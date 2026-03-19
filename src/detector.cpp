#include "detector.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "ocr_common.h"

namespace rapidocr {
namespace {

constexpr int kBoxSortYThreshold = 10;

std::vector<cv::Point2f> OrderPointsClockwise(const std::vector<cv::Point>& pts) {
    std::vector<cv::Point2f> points;
    points.reserve(pts.size());
    for (const auto& p : pts) {
        points.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
    }

    std::sort(points.begin(), points.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        if (a.x != b.x) {
            return a.x < b.x;
        }
        return a.y < b.y;
    });

    std::vector<cv::Point2f> leftMost(points.begin(), points.begin() + 2);
    std::vector<cv::Point2f> rightMost(points.begin() + 2, points.end());

    std::sort(leftMost.begin(), leftMost.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;
    });
    std::sort(rightMost.begin(), rightMost.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;
    });

    const cv::Point2f tl = leftMost[0];
    const cv::Point2f bl = leftMost[1];
    const cv::Point2f tr = rightMost[0];
    const cv::Point2f br = rightMost[1];

    return {tl, tr, br, bl};
}

std::vector<cv::Point> ClipDetRes(
    const std::vector<cv::Point2f>& points,
    int imgHeight,
    int imgWidth) {
    std::vector<cv::Point> out;
    out.reserve(points.size());
    for (const auto& p : points) {
        out.emplace_back(
            ClampInt(static_cast<int>(p.x), 0, imgWidth - 1),
            ClampInt(static_cast<int>(p.y), 0, imgHeight - 1));
    }
    return out;
}

int NormL2Int(const cv::Point& a, const cv::Point& b) {
    const float dx = static_cast<float>(a.x - b.x);
    const float dy = static_cast<float>(a.y - b.y);
    return static_cast<int>(std::sqrt(dx * dx + dy * dy));
}

std::vector<TextBox> FilterDetRes(
    const std::vector<TextBox>& textBoxes,
    int imgHeight,
    int imgWidth,
    bool mergeCodeLines) {
    std::vector<TextBox> filtered;
    filtered.reserve(textBoxes.size());

    for (const auto& textBox : textBoxes) {
        if (textBox.boxPoints.size() != 4) {
            continue;
        }

        const std::vector<cv::Point2f> ordered = OrderPointsClockwise(textBox.boxPoints);
        const std::vector<cv::Point> clipped = ClipDetRes(ordered, imgHeight, imgWidth);

        const int rectWidth = NormL2Int(clipped[0], clipped[1]);
        const int rectHeight = NormL2Int(clipped[0], clipped[3]);

        if (mergeCodeLines) {
            const int area = rectWidth * rectHeight;
            if ((rectWidth <= 2 && rectHeight <= 2) || area <= 4) {
                continue;
            }
        } else {
            if (rectWidth <= 3 || rectHeight <= 3) {
                continue;
            }
        }

        filtered.push_back(TextBox{clipped, textBox.score});
    }
    return filtered;
}

std::vector<TextBox> SortBoxesLikePython(const std::vector<TextBox>& boxes) {
    if (boxes.empty()) {
        return boxes;
    }

    std::vector<TextBox> ySorted = boxes;
    std::stable_sort(ySorted.begin(), ySorted.end(), [](const TextBox& a, const TextBox& b) {
        return a.boxPoints[0].y < b.boxPoints[0].y;
    });

    std::vector<int> lineIds(ySorted.size(), 0);
    for (std::size_t i = 1; i < ySorted.size(); ++i) {
        const int dy = ySorted[i].boxPoints[0].y - ySorted[i - 1].boxPoints[0].y;
        lineIds[i] = lineIds[i - 1] + ((dy >= kBoxSortYThreshold) ? 1 : 0);
    }

    std::vector<std::size_t> order(ySorted.size());
    std::iota(order.begin(), order.end(), 0);

    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (lineIds[a] != lineIds[b]) {
            return lineIds[a] < lineIds[b];
        }
        return ySorted[a].boxPoints[0].x < ySorted[b].boxPoints[0].x;
    });

    std::vector<TextBox> out;
    out.reserve(ySorted.size());
    for (std::size_t idx : order) {
        out.push_back(ySorted[idx]);
    }
    return out;
}

struct BoxRect {
    int left;
    int top;
    int right;
    int bottom;
    int width;
    int height;
    int centerY;
};

BoxRect GetBoxRect(const TextBox& box) {
    BoxRect rect{};

    if (box.boxPoints.empty()) {
        return rect;
    }

    int minX = box.boxPoints[0].x;
    int minY = box.boxPoints[0].y;
    int maxX = box.boxPoints[0].x;
    int maxY = box.boxPoints[0].y;

    for (std::size_t i = 1; i < box.boxPoints.size(); ++i) {
        minX = std::min(minX, box.boxPoints[i].x);
        minY = std::min(minY, box.boxPoints[i].y);
        maxX = std::max(maxX, box.boxPoints[i].x);
        maxY = std::max(maxY, box.boxPoints[i].y);
    }

    rect.left = minX;
    rect.top = minY;
    rect.right = maxX;
    rect.bottom = maxY;
    rect.width = std::max(0, maxX - minX);
    rect.height = std::max(0, maxY - minY);
    rect.centerY = (minY + maxY) / 2;
    return rect;
}

bool IsLikelySameCodeLine(const TextBox& aBox, const TextBox& bBox) {
    const BoxRect a = GetBoxRect(aBox);
    const BoxRect b = GetBoxRect(bBox);

    if (a.width <= 0 || a.height <= 0 || b.width <= 0 || b.height <= 0) {
        return false;
    }

    const int minHeight = std::min(a.height, b.height);
    const int maxHeight = std::max(a.height, b.height);
    if (maxHeight > std::max(2, minHeight * 2)) {
        return false;
    }

    const int centerYDiff = std::abs(a.centerY - b.centerY);
    const int topDiff = std::abs(a.top - b.top);
    const int bottomDiff = std::abs(a.bottom - b.bottom);

    const int centerYThresh = std::max(4, minHeight / 2);
    const int edgeYThresh = std::max(6, maxHeight / 2);

    if (centerYDiff > centerYThresh) {
        return false;
    }
    if (topDiff > edgeYThresh || bottomDiff > edgeYThresh) {
        return false;
    }

    const int overlapTop = std::max(a.top, b.top);
    const int overlapBottom = std::min(a.bottom, b.bottom);
    const int overlapH = overlapBottom - overlapTop;
    if (overlapH < std::max(2, minHeight / 3)) {
        return false;
    }

    return true;
}

TextBox MergeTextBoxes(const std::vector<TextBox>& boxes) {
    TextBox merged;
    if (boxes.empty()) {
        return merged;
    }

    BoxRect first = GetBoxRect(boxes[0]);
    int left = first.left;
    int top = first.top;
    int right = first.right;
    int bottom = first.bottom;
    float score = boxes[0].score;

    for (std::size_t i = 1; i < boxes.size(); ++i) {
        const BoxRect rect = GetBoxRect(boxes[i]);
        left = std::min(left, rect.left);
        top = std::min(top, rect.top);
        right = std::max(right, rect.right);
        bottom = std::max(bottom, rect.bottom);
        score = std::max(score, boxes[i].score);
    }

    merged.boxPoints.push_back(cv::Point(left, top));
    merged.boxPoints.push_back(cv::Point(right, top));
    merged.boxPoints.push_back(cv::Point(right, bottom));
    merged.boxPoints.push_back(cv::Point(left, bottom));
    merged.score = score;
    return merged;
}

std::vector<std::vector<TextBox> > GroupBoxesIntoTextLines(
    const std::vector<TextBox>& sortedBoxes) {

    std::vector<std::vector<TextBox> > lines;
    if (sortedBoxes.empty()) {
        return lines;
    }

    for (const auto& box : sortedBoxes) {
        bool assigned = false;

        for (std::size_t i = 0; i < lines.size(); ++i) {
            if (lines[i].empty()) {
                continue;
            }

			if (IsLikelySameCodeLine(lines[i].back(), box)) {
				lines[i].push_back(box);
				assigned = true;
				break;
			}
        }

        if (!assigned) {
            lines.push_back(std::vector<TextBox>{box});
        }
    }

    for (std::size_t i = 0; i < lines.size(); ++i) {
        std::stable_sort(lines[i].begin(), lines[i].end(),
            [](const TextBox& a, const TextBox& b) {
                return GetBoxRect(a).left < GetBoxRect(b).left;
            });
    }

    return lines;
}

std::vector<TextBox> MergeBoxesForCodeLines(
    const std::vector<TextBox>& sortedBoxes) {

    if (sortedBoxes.empty()) {
        return {};
    }

    const std::vector<std::vector<TextBox> > lines =
        GroupBoxesIntoTextLines(sortedBoxes);

    std::vector<TextBox> merged;
    merged.reserve(lines.size());

    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (!lines[i].empty()) {
            merged.push_back(MergeTextBoxes(lines[i]));
        }
    }

    return merged;
}


float ResolveDetLimitSideLen(const cv::Mat& src, const OcrRunOptions& options) {
    if (options.limitType == "min") {
        return options.limitSideLen;
    }

    const int maxWh = std::max(src.rows, src.cols);
    if (maxWh < 960) {
        return 960.0f;
    }
    if (maxWh < 1500) {
        return 1500.0f;
    }
    return 2000.0f;
}

std::vector<TextBox> FindResultBoxes(
    const cv::Mat& predMat,
    const cv::Mat& maskMat,
    const ScaleParam& scaleParam,
    float boxScoreThresh,
    int maxCandidates,
    float unclipRatio,
    const std::string& scoreMode,
    bool mergeCodeLines) {
    const int kMinLongSide = mergeCodeLines ? 2 : 3;
    const int kMinUnclipShortSide = mergeCodeLines ? 3 : (kMinLongSide + 2);

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(maskMat, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    const std::size_t contourCount = std::min<std::size_t>(
        contours.size(), static_cast<std::size_t>(std::max(0, maxCandidates)));
    std::vector<TextBox> result;
    result.reserve(contourCount);

    for (std::size_t i = 0; i < contourCount; ++i) {
        if (contours[i].size() <= 2) {
            continue;
        }

        const cv::RotatedRect minAreaRect = cv::minAreaRect(contours[i]);

        float shortSide = 0.0f;
        const std::vector<cv::Point2f> minBoxes = GetMinBoxes(minAreaRect, shortSide);
        if (shortSide < kMinLongSide) {
            continue;
        }

        const float boxScore = (scoreMode == "slow")
            ? BoxScoreSlow(contours[i], predMat)
            : BoxScoreFast(minBoxes, predMat);
        if (boxScoreThresh > boxScore) {
            continue;
        }

        const cv::RotatedRect clipRect = Unclip(minBoxes, unclipRatio);
        if (clipRect.size.height < 1.001f && clipRect.size.width < 1.001f) {
            continue;
        }

        const std::vector<cv::Point2f> clipMinBoxes = GetMinBoxes(clipRect, shortSide);
        if (shortSide < kMinUnclipShortSide) {
            continue;
        }

        std::vector<cv::Point> mappedPoints;
        mappedPoints.reserve(4);
        for (const cv::Point2f& point : clipMinBoxes) {
            const int x = ClampInt(
                RoundHalfToEven(
                    point.x / static_cast<float>(predMat.cols) *
                    static_cast<float>(scaleParam.srcWidth)),
                0,
                scaleParam.srcWidth);

            const int y = ClampInt(
                RoundHalfToEven(
                    point.y / static_cast<float>(predMat.rows) *
                    static_cast<float>(scaleParam.srcHeight)),
                0,
                scaleParam.srcHeight);

            mappedPoints.emplace_back(x, y);
        }

        result.push_back(TextBox{mappedPoints, boxScore});
    }

    result = FilterDetRes(
        result,
        scaleParam.srcHeight,
        scaleParam.srcWidth,
        mergeCodeLines);
    result = SortBoxesLikePython(result);

    if (mergeCodeLines) {
        result = MergeBoxesForCodeLines(result);
    }
    return result;
}

}  // namespace

Detector::Detector()
    : env_(ORT_LOGGING_LEVEL_ERROR, "Detector"),
      meanValues_{0.5f, 0.5f, 0.5f},
      normValues_{0.5f, 0.5f, 0.5f} {}

void Detector::ConfigureSessionOptions() {
    sessionOptions_ = Ort::SessionOptions();
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

void Detector::Initialize(const std::string& modelPath) {
    if (modelPath.empty()) {
        throw std::invalid_argument("detector model path is empty");
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

void Detector::ValidateReady() const {
    if (!session_) {
        throw std::runtime_error("text detector is not initialized");
    }
    if (inputNames_.empty() || outputNames_.empty()) {
        throw std::runtime_error("text detector input/output metadata is empty");
    }
}

std::vector<TextBox> Detector::Detect(const cv::Mat& src, const OcrRunOptions& options) const {
    ValidateReady();
    if (src.empty()) {
        return {};
    }

    const float detLimitSideLen = ResolveDetLimitSideLen(src, options);
    const ScaleParam scaleParam = GetDetScaleParam(src, detLimitSideLen, options.limitType);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(scaleParam.dstWidth, scaleParam.dstHeight));

    std::vector<float> inputTensorValues = SubtractMeanNormalize(resized, meanValues_, normValues_);
    const std::array<int64_t, 4> inputShape = {1, resized.channels(), resized.rows, resized.cols};

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
        Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

    if (outputTensor.size() != 1 || !outputTensor.front().IsTensor()) {
        throw std::runtime_error("unexpected detector output tensor");
    }

    const std::vector<int64_t> outputShape =
        outputTensor.front().GetTensorTypeAndShapeInfo().GetShape();
    if (outputShape.size() < 4) {
        throw std::runtime_error("invalid detector output shape");
    }

    const int outHeight = static_cast<int>(outputShape[2]);
    const int outWidth = static_cast<int>(outputShape[3]);
    const float* outputData = outputTensor.front().GetTensorData<float>();
    const std::size_t area =
        static_cast<std::size_t>(outHeight) * static_cast<std::size_t>(outWidth);

    cv::Mat predMat(outHeight, outWidth, CV_32F, const_cast<float*>(outputData));

    cv::Mat mask(outHeight, outWidth, CV_8UC1);
    for (std::size_t i = 0; i < area; ++i) {
        mask.data[i] = (outputData[i] > options.thresh) ? 255 : 0;
    }

	if (options.useDilation) {
		cv::Mat dilated;
		const cv::Mat element = cv::Mat::ones(2, 2, CV_8UC1);
		cv::dilate(mask, dilated, element);
		mask = dilated;
	}

	return FindResultBoxes(
		predMat,
		mask,
		scaleParam,
		options.boxThresh,
		options.maxCandidates,
		options.unclipRatio,
		options.scoreMode,
		options.mergeCodeLines);
}

}  // namespace rapidocr