#include "recognizer.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "ocr_common.h"

namespace rapidocr {
namespace {

constexpr int kScoreRoundDigits = 5;

inline float RoundScore5(float value) {
    return static_cast<float>(RoundHalfToEven(static_cast<double>(value), kScoreRoundDigits));
}

std::vector<std::string> SplitLines(const std::string& raw) {
    std::vector<std::string> out;
    std::stringstream ss(raw);
    std::string line;
    while (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            out.push_back(line);
        }
    }
    return out;
}

}  // namespace

Recognizer::Recognizer()
    : env_(ORT_LOGGING_LEVEL_ERROR, "Recognizer") {}

void Recognizer::ConfigureSessionOptions() {
    sessionOptions_ = Ort::SessionOptions();
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

std::vector<std::string> Recognizer::SplitCharacterList(const std::string& raw) {
    return SplitLines(raw);
}

std::vector<std::string> Recognizer::LoadKeysFromModel() const {
    if (!session_) {
        throw std::runtime_error("text recognizer is not initialized");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::ModelMetadata metadata = session_->GetModelMetadata();

    const char* candidateKeys[] = {
        "character",
        "character_list",
        "keys",
        "ocr_character",
        "rec_character"
    };

    for (std::size_t i = 0; i < sizeof(candidateKeys) / sizeof(candidateKeys[0]); ++i) {
        try {
            Ort::AllocatedStringPtr value =
                metadata.LookupCustomMetadataMapAllocated(candidateKeys[i], allocator);
            if (value && value.get() != NULL && value.get()[0] != '\0') {
                std::vector<std::string> chars = SplitCharacterList(value.get());
                if (!chars.empty()) {
                    return chars;
                }
            }
        } catch (...) {
        }
    }

    throw std::runtime_error("failed to read built-in character list from recognizer model");
}

void Recognizer::Initialize(const std::string& modelPath) {
    if (modelPath.empty()) {
        throw std::invalid_argument("recognizer model path is empty");
    }

    ConfigureSessionOptions();

#ifdef _WIN32
    const std::wstring widePath = Utf8ToWide(modelPath);
    session_ = std::make_unique<Ort::Session>(env_, widePath.c_str(), sessionOptions_);
#else
    session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions_);
#endif

    inputNames_ = GetInputNames(*session_);
    outputNames_ = GetOutputNames(*session_);

    keys_ = LoadKeysFromModel();
    if (keys_.empty()) {
        throw std::runtime_error("built-in recognizer keys are empty");
    }

    keys_.push_back(" ");
    keys_.insert(keys_.begin(), "blank");
}

void Recognizer::ValidateReady() const {
    if (!session_) {
        throw std::runtime_error("text recognizer is not initialized");
    }
    if (inputNames_.empty() || outputNames_.empty()) {
        throw std::runtime_error("text recognizer input/output metadata is empty");
    }
    if (keys_.empty()) {
        throw std::runtime_error("text recognizer keys are not initialized");
    }
}

std::vector<float> Recognizer::ResizeNormImg(
    const cv::Mat& img,
    float maxWhRatio) const {
    if (img.empty()) {
        return std::vector<float>(
            static_cast<std::size_t>(imgChannels_ * imgHeight_ * imgWidth_), 0.0f);
    }
    if (img.channels() != imgChannels_) {
        throw std::runtime_error("recognizer expects 3-channel image");
    }

    int targetWidth = static_cast<int>(imgHeight_ * maxWhRatio);

    const int h = img.rows;
    const int w = img.cols;
    const float ratio = static_cast<float>(w) / static_cast<float>(h);

    int resizedW = 0;
    if (static_cast<int>(std::ceil(imgHeight_ * ratio)) > targetWidth) {
        resizedW = targetWidth;
    } else {
        resizedW = static_cast<int>(std::ceil(imgHeight_ * ratio));
    }
    resizedW = std::max(resizedW, 1);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resizedW, imgHeight_));

    std::vector<float> paddingIm(
        static_cast<std::size_t>(imgChannels_ * imgHeight_ * targetWidth), 0.0f);
    const int imageSize = imgHeight_ * targetWidth;

    for (int y = 0; y < imgHeight_; ++y) {
        const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < resizedW; ++x) {
            for (int c = 0; c < imgChannels_; ++c) {
                float v = static_cast<float>(row[x][c]);
                v = v / 255.0f;
                v -= 0.5f;
                v /= 0.5f;
                paddingIm[c * imageSize + y * targetWidth + x] = v;
            }
        }
    }

    return paddingIm;
}

TextLine Recognizer::DecodeTextLine(
    const float* outputData,
    std::size_t steps,
    std::size_t classes) const {
    std::string text;
    text.reserve(steps);
    std::vector<float> scores;
    scores.reserve(steps);
    
    std::size_t lastIndex = 0;

    for (std::size_t step = 0; step < steps; ++step) {
        const float* begin = outputData + step * classes;
        const float* maxIt = std::max_element(begin, begin + classes);
        const std::size_t maxIndex = static_cast<std::size_t>(maxIt - begin);

        if (maxIndex == 0 || maxIndex == lastIndex || maxIndex >= keys_.size()) {
            lastIndex = maxIndex;
            continue;
        }

        text.append(keys_[maxIndex]);
        scores.push_back(RoundScore5(*maxIt));
        lastIndex = maxIndex;
    }

    return {text, scores};
}

std::vector<TextLine> Recognizer::Recognize(
    const std::vector<cv::Mat>& partImages) const {
    ValidateReady();

    std::vector<TextLine> results(partImages.size());
    if (partImages.empty()) {
        return results;
    }

    std::vector<RatioIndex> order;
    order.reserve(partImages.size());
    for (int i = 0; i < static_cast<int>(partImages.size()); ++i) {
        const cv::Mat& img = partImages[i];
        const float ratio = img.empty()
            ? 0.0f
            : static_cast<float>(img.cols) / static_cast<float>(img.rows);
        order.push_back(RatioIndex{i, ratio});
    }

    std::sort(
        order.begin(),
        order.end(),
        [](const RatioIndex& a, const RatioIndex& b) {
            return a.ratio < b.ratio;
        });

    const char* inputNames[] = {inputNames_.front().get()};
    const char* outputNames[] = {outputNames_.front().get()};
    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (std::size_t beg = 0; beg < order.size(); beg += static_cast<std::size_t>(recBatchNum_)) {
        const std::size_t end =
            std::min(beg + static_cast<std::size_t>(recBatchNum_), order.size());
        const std::size_t batchSize = end - beg;

		float maxWhRatio =
			static_cast<float>(imgWidth_) / static_cast<float>(imgHeight_);

		for (std::size_t i = beg; i < end; ++i) {
			const cv::Mat& img = partImages[order[i].index];
			const float whRatio = static_cast<float>(img.cols) / static_cast<float>(img.rows);
			maxWhRatio = std::max(maxWhRatio, whRatio);
		}

        const int batchWidth = static_cast<int>(imgHeight_ * maxWhRatio);
        const int singleSize = imgChannels_ * imgHeight_ * batchWidth;

        std::vector<float> inputTensorValues(
            static_cast<std::size_t>(singleSize) * batchSize, 0.0f);

        for (std::size_t i = 0; i < batchSize; ++i) {
            const cv::Mat& img = partImages[order[beg + i].index];
            std::vector<float> norm = ResizeNormImg(img, maxWhRatio);
            std::copy(
                norm.begin(),
                norm.end(),
                inputTensorValues.begin() + static_cast<std::ptrdiff_t>(i * singleSize));
        }

        const int64_t inputShape[4] = {
            static_cast<int64_t>(batchSize),
            static_cast<int64_t>(imgChannels_),
            static_cast<int64_t>(imgHeight_),
            static_cast<int64_t>(batchWidth)
        };

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensorValues.data(),
            inputTensorValues.size(),
            inputShape,
            4);

        std::vector<Ort::Value> outputTensor = session_->Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1);

        if (outputTensor.size() != 1 || !outputTensor.front().IsTensor()) {
            throw std::runtime_error("unexpected recognizer output tensor");
        }

        const std::vector<int64_t> outputShape =
            outputTensor.front().GetTensorTypeAndShapeInfo().GetShape();
        if (outputShape.size() != 3) {
            throw std::runtime_error("invalid recognizer output shape, expect [N, T, C]");
        }

        const std::size_t n = static_cast<std::size_t>(outputShape[0]);
        const std::size_t t = static_cast<std::size_t>(outputShape[1]);
        const std::size_t c = static_cast<std::size_t>(outputShape[2]);
        if (n != batchSize) {
            throw std::runtime_error("recognizer output batch size mismatch");
        }

        const float* outputData = outputTensor.front().GetTensorData<float>();
        const std::size_t oneSize = t * c;

        for (std::size_t i = 0; i < batchSize; ++i) {
            TextLine line = DecodeTextLine(outputData + i * oneSize, t, c);
            results[order[beg + i].index] = line;
        }
    }

    return results;
}

}  // namespace rapidocr