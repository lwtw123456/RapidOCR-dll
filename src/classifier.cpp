#include "classifier.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <opencv2/imgproc.hpp>

#include "ocr_common.h"

namespace rapidocr {
namespace {

AnglePrediction ScoreToAngle(const float* outputData, std::size_t count) {
    int maxIndex = 0;
    float maxScore = count > 0 ? outputData[0] : 0.0f;
    for (std::size_t i = 1; i < count; ++i) {
        if (outputData[i] > maxScore) {
            maxScore = outputData[i];
            maxIndex = static_cast<int>(i);
        }
    }
    return {maxIndex, maxScore};
}

}  // namespace

Classifier::Classifier()
    : env_(ORT_LOGGING_LEVEL_ERROR, "Classifier"),
      meanValues_{0.5f, 0.5f, 0.5f},
      normValues_{0.5f, 0.5f, 0.5f},
      labelList_{"0", "180"} {}

void Classifier::ConfigureSessionOptions() {
    sessionOptions_ = Ort::SessionOptions();
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

void Classifier::Initialize(const std::string& modelPath) {
    if (modelPath.empty()) {
        throw std::invalid_argument("classifier model path is empty");
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
}

void Classifier::ValidateReady() const {
    if (!session_) {
        throw std::runtime_error("text classifier is not initialized");
    }
    if (inputNames_.empty() || outputNames_.empty()) {
        throw std::runtime_error("text classifier input/output metadata is empty");
    }
}

cv::Mat Classifier::ResizeNormImg(const cv::Mat& img) const {
    if (img.empty()) {
        throw std::runtime_error("classifier input image is empty");
    }

    const int imgC = 3;
    const int imgH = dstHeight_;
    const int imgW = dstWidth_;
    const float ratio = static_cast<float>(img.cols) / static_cast<float>(img.rows);
    const int resizedW = std::min(imgW, static_cast<int>(std::ceil(imgH * ratio)));

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resizedW, imgH));
    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

    const std::size_t planeSize = static_cast<std::size_t>(imgH) * imgW;
    std::vector<float> output(imgC * planeSize, 0.0f);

    const float* srcPtr = reinterpret_cast<const float*>(resized.data);
    const std::size_t srcStep = resized.step1();
    
    for (int y = 0; y < imgH; ++y) {
        const float* srcRow = srcPtr + y * srcStep;
        for (int x = 0; x < resizedW; ++x) {
            const std::size_t srcIdx = x * 3;
            const std::size_t dstIdx = y * imgW + x;
            output[dstIdx] = (srcRow[srcIdx] - meanValues_[0]) / normValues_[0];
            output[planeSize + dstIdx] = (srcRow[srcIdx + 1] - meanValues_[1]) / normValues_[1];
            output[2 * planeSize + dstIdx] = (srcRow[srcIdx + 2] - meanValues_[2]) / normValues_[2];
        }
    }

    return cv::Mat({imgC, imgH, imgW}, CV_32F, output.data()).clone();
}

std::vector<AnglePrediction> Classifier::PredictBatch(const std::vector<cv::Mat>& batchImages) const {
    if (batchImages.empty()) {
        return {};
    }

    const int batchSize = static_cast<int>(batchImages.size());
    const int imgC = 3;
    const int imgH = dstHeight_;
    const int imgW = dstWidth_;
    const std::size_t singleImageSize =
        static_cast<std::size_t>(imgC) * static_cast<std::size_t>(imgH) * static_cast<std::size_t>(imgW);

    std::vector<float> inputTensorValues(static_cast<std::size_t>(batchSize) * singleImageSize);

    for (int i = 0; i < batchSize; ++i) {
        cv::Mat normImg = ResizeNormImg(batchImages[i]);
        std::memcpy(
            inputTensorValues.data() + static_cast<std::size_t>(i) * singleImageSize,
            normImg.ptr<float>(),
            singleImageSize * sizeof(float));
    }

    const std::array<int64_t, 4> inputShape = {batchSize, imgC, imgH, imgW};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensorValues.data(),
        inputTensorValues.size(),
        inputShape.data(),
        inputShape.size());

    const char* inputNames[] = {inputNames_.front().get()};
    const char* outputNames[] = {outputNames_.front().get()};

    std::vector<Ort::Value> outputTensor = session_->Run(
        Ort::RunOptions{nullptr},
        inputNames,
        &inputTensor,
        1,
        outputNames,
        1);

    if (outputTensor.size() != 1 || !outputTensor.front().IsTensor()) {
        throw std::runtime_error("unexpected classifier output tensor");
    }

    auto shapeInfo = outputTensor.front().GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> outputShape = shapeInfo.GetShape();

    if (outputShape.size() != 2) {
        throw std::runtime_error("classifier output shape must be [N, C]");
    }

    const int64_t outBatch = outputShape[0];
    const int64_t outCols = outputShape[1];
    if (outBatch != batchSize || outCols <= 0) {
        throw std::runtime_error("classifier output shape mismatch");
    }

    const float* outputData = outputTensor.front().GetTensorData<float>();
    std::vector<AnglePrediction> results(batchSize);

    for (int i = 0; i < batchSize; ++i) {
        results[i] = ScoreToAngle(
            outputData + static_cast<std::size_t>(i) * static_cast<std::size_t>(outCols),
            static_cast<std::size_t>(outCols));
    }

    return results;
}

std::vector<AnglePrediction> Classifier::Predict(
    const std::vector<cv::Mat>& partImages) const {
    ValidateReady();

    std::vector<AnglePrediction> predictions(partImages.size());
    if (partImages.empty()) {
        return predictions;
    }

    std::vector<float> widthList(partImages.size(), 0.0f);
    for (std::size_t i = 0; i < partImages.size(); ++i) {
		if (partImages[i].empty() || partImages[i].rows <= 0) {
			widthList[i] = 0.0f;
		} else {
			widthList[i] = static_cast<float>(partImages[i].cols) /
						   static_cast<float>(partImages[i].rows);
		}
	}

    std::vector<int> indices(partImages.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return widthList[a] < widthList[b]; });

    for (std::size_t beg = 0; beg < partImages.size();
         beg += static_cast<std::size_t>(clsBatchNum_)) {
        const std::size_t end =
            std::min(partImages.size(), beg + static_cast<std::size_t>(clsBatchNum_));

        std::vector<cv::Mat> batchImages;
        batchImages.reserve(end - beg);

        for (std::size_t i = beg; i < end; ++i) {
            batchImages.push_back(partImages[indices[i]]);
        }

        std::vector<AnglePrediction> batchResults = PredictBatch(batchImages);

        for (std::size_t r = 0; r < batchResults.size(); ++r) {
            predictions[indices[beg + r]] = batchResults[r];
        }
    }

    return predictions;
}

bool Classifier::ShouldRotate180(const AnglePrediction& prediction) const {
    if (prediction.index < 0 ||
        prediction.index >= static_cast<int>(labelList_.size())) {
        return false;
    }
    return labelList_[static_cast<std::size_t>(prediction.index)] == "180" &&
           prediction.score > clsThresh_;
}

}  // namespace rapidocr