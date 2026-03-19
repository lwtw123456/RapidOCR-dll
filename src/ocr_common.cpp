#include "ocr_common.h"

#ifdef _WIN32
#include <Windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <sys/stat.h>

#include <opencv2/imgproc.hpp>

#include "clipper.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace rapidocr {

int ClampInt(int value, int minValue, int maxValue) {
    return std::max(minValue, std::min(value, maxValue));
}

int RoundHalfToEven(float value) {
    const float floorValue = std::floor(value);
    const float diff = value - floorValue;

    if (diff < 0.5f) {
        return static_cast<int>(floorValue);
    }
    if (diff > 0.5f) {
        return static_cast<int>(floorValue + 1.0f);
    }

    const int floorInt = static_cast<int>(floorValue);
    return (floorInt % 2 == 0) ? floorInt : (floorInt + 1);
}

double RoundHalfToEven(double value, int digits) {
    if (digits < 0) {
        return value;
    }

    double scale = 1.0;
    for (int i = 0; i < digits; ++i) {
        scale *= 10.0;
    }

    const double scaled = value * scale;
    const double floorValue = std::floor(scaled);
    const double diff = scaled - floorValue;

    double rounded = 0.0;
    if (diff < 0.5) {
        rounded = floorValue;
    } else if (diff > 0.5) {
        rounded = floorValue + 1.0;
    } else {
        const long long floorInt = static_cast<long long>(floorValue);
        rounded = (floorInt % 2 == 0) ? floorValue : (floorValue + 1.0);
    }

    return rounded / scale;
}

namespace {

bool ComparePointX(const cv::Point2f& a, const cv::Point2f& b) {
    return a.x < b.x;
}

float GetContourArea(const std::vector<cv::Point2f>& box, float unclipRatio) {
    const std::size_t size = box.size();
    float area = 0.0f;
    float perimeter = 0.0f;
    for (std::size_t i = 0; i < size; ++i) {
        area += box[i].x * box[(i + 1) % size].y - box[i].y * box[(i + 1) % size].x;
        const float dx = box[i].x - box[(i + 1) % size].x;
        const float dy = box[i].y - box[(i + 1) % size].y;
        perimeter += std::sqrt(dx * dx + dy * dy);
    }
    area = std::fabs(area / 2.0f);
    if (perimeter <= 0.0f) {
        return 0.0f;
    }
    return area * unclipRatio / perimeter;
}

std::vector<cv::Point2f> GetBox(const cv::RotatedRect& rect) {
    cv::Point2f vertices[4];
    rect.points(vertices);
    return std::vector<cv::Point2f>(vertices, vertices + 4);
}

}  // namespace

bool FileExists(const std::string& path) {
    struct stat buffer;
    return stat(path.c_str(), &buffer) == 0;
}

std::wstring Utf8ToWide(const std::string& value) {
    if (value.empty()) {
        return L"";
    }

#ifdef _WIN32
    const int sizeNeeded = MultiByteToWideChar(
        CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), NULL, 0);
    if (sizeNeeded <= 0) {
        return L"";
    }

    std::wstring wide(static_cast<std::size_t>(sizeNeeded), L'\0');
    MultiByteToWideChar(
        CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), &wide[0], sizeNeeded);
    return wide;
#else
    return std::wstring(value.begin(), value.end());
#endif
}

ScaleParam GetScaleParam(const cv::Mat& src, float scale) {
    const float dstWidthF = static_cast<float>(src.cols) * scale;
    const float dstHeightF = static_cast<float>(src.rows) * scale;

    const int dstWidth = std::max(RoundHalfToEven(dstWidthF / 32.0f) * 32, 32);
    const int dstHeight = std::max(RoundHalfToEven(dstHeightF / 32.0f) * 32, 32);

    return {src.cols, src.rows, dstWidth, dstHeight,
            static_cast<float>(dstWidth) / static_cast<float>(src.cols),
            static_cast<float>(dstHeight) / static_cast<float>(src.rows)};
}

ScaleParam GetScaleParam(const cv::Mat& src, int targetSize) {
    const float ratio = src.cols > src.rows
                            ? static_cast<float>(targetSize) / static_cast<float>(src.cols)
                            : static_cast<float>(targetSize) / static_cast<float>(src.rows);

    const float dstWidthF = static_cast<float>(src.cols) * ratio;
    const float dstHeightF = static_cast<float>(src.rows) * ratio;

    const int dstWidth = std::max(RoundHalfToEven(dstWidthF / 32.0f) * 32, 32);
    const int dstHeight = std::max(RoundHalfToEven(dstHeightF / 32.0f) * 32, 32);

    return {src.cols, src.rows, dstWidth, dstHeight,
            static_cast<float>(dstWidth) / static_cast<float>(src.cols),
            static_cast<float>(dstHeight) / static_cast<float>(src.rows)};
}

ScaleParam GetDetScaleParam(const cv::Mat& src, float limitSideLen, const std::string& limitType) {
    if (src.empty()) {
        return {};
    }

    const int h = src.rows;
    const int w = src.cols;
    float ratio = 1.0f;

    if (limitType == "max") {
        const int maxSide = std::max(h, w);
        if (static_cast<float>(maxSide) > limitSideLen) {
            ratio = limitSideLen / static_cast<float>(maxSide);
        }
    } else {
        const int minSide = std::min(h, w);
        if (static_cast<float>(minSide) < limitSideLen) {
            ratio = limitSideLen / static_cast<float>(minSide);
        }
    }

    int resizeH = static_cast<int>(static_cast<float>(h) * ratio);
    int resizeW = static_cast<int>(static_cast<float>(w) * ratio);

    resizeH = std::max(RoundHalfToEven(static_cast<float>(resizeH) / 32.0f) * 32, 32);
    resizeW = std::max(RoundHalfToEven(static_cast<float>(resizeW) / 32.0f) * 32, 32);

    return {w, h, resizeW, resizeH,
            static_cast<float>(resizeW) / static_cast<float>(w),
            static_cast<float>(resizeH) / static_cast<float>(h)};
}

cv::Mat ResizeBySideLimit(const cv::Mat& src, int minSideLen, int maxSideLen, float& scale) {
    scale = 1.0f;
    if (src.empty()) {
        return cv::Mat();
    }

    const int srcH = src.rows;
    const int srcW = src.cols;
    const int minSide = std::min(srcH, srcW);
    const int maxSide = std::max(srcH, srcW);

    if (minSideLen > 0 && minSide < minSideLen) {
        scale = static_cast<float>(minSideLen) / static_cast<float>(minSide);
    }
    if (maxSideLen > 0 && maxSide * scale > maxSideLen) {
        scale = static_cast<float>(maxSideLen) / static_cast<float>(maxSide);
    }

    if (std::fabs(scale - 1.0f) < 1e-6f) {
        scale = 1.0f;
        return src;
    }

    const int dstW = std::max(1, RoundHalfToEven(static_cast<float>(srcW) * scale));
    const int dstH = std::max(1, RoundHalfToEven(static_cast<float>(srcH) * scale));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(dstW, dstH));
    scale = static_cast<float>(dstW) / static_cast<float>(srcW);
    return resized;
}

int GetThickness(const cv::Mat& image) {
    const int minSize = std::min(image.cols, image.rows);
    return minSize / 1000 + 2;
}

void DrawTextBoxes(cv::Mat& image, const std::vector<TextBox>& textBoxes, int thickness) {
    const cv::Scalar color(0, 0, 255);
    for (const TextBox& textBox : textBoxes) {
        if (textBox.boxPoints.size() != 4) {
            continue;
        }
        for (int i = 0; i < 4; ++i) {
            cv::line(image, textBox.boxPoints[i], textBox.boxPoints[(i + 1) % 4], color, thickness);
        }
    }
}

cv::Mat Rotate180(const cv::Mat& src) {
    cv::Mat out;
    cv::flip(src, out, 0);
    cv::flip(out, out, 1);
    return out;
}

cv::Mat GetRotateCropImage(const cv::Mat& src, const std::vector<cv::Point>& box) {
    if (box.size() != 4) {
        throw std::invalid_argument("text box must contain exactly 4 points");
    }

    int minX = box[0].x, maxX = box[0].x;
    int minY = box[0].y, maxY = box[0].y;
    
    for (int i = 1; i < 4; ++i) {
        minX = std::min(minX, box[i].x);
        maxX = std::max(maxX, box[i].x);
        minY = std::min(minY, box[i].y);
        maxY = std::max(maxY, box[i].y);
    }
    
    minX = std::max(0, minX);
    maxX = std::min(src.cols - 1, maxX);
    minY = std::max(0, minY);
    maxY = std::min(src.rows - 1, maxY);

    if (maxX <= minX || maxY <= minY) return cv::Mat();

    cv::Mat cropped = src(cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1)).clone();
    
    cv::Point2f srcPts[4], dst[4];
    for (int i = 0; i < 4; ++i) {
        srcPts[i] = cv::Point2f(box[i].x - minX, box[i].y - minY);
    }

    const float dx01 = srcPts[0].x - srcPts[1].x;
    const float dy01 = srcPts[0].y - srcPts[1].y;
    const float dx03 = srcPts[0].x - srcPts[3].x;
    const float dy03 = srcPts[0].y - srcPts[3].y;
    
    const int cropWidth = static_cast<int>(std::sqrt(dx01 * dx01 + dy01 * dy01));
    const int cropHeight = static_cast<int>(std::sqrt(dx03 * dx03 + dy03 * dy03));

    if (cropWidth <= 0 || cropHeight <= 0) return cv::Mat();

    dst[0] = cv::Point2f(0.0f, 0.0f);
    dst[1] = cv::Point2f(cropWidth, 0.0f);
    dst[2] = cv::Point2f(cropWidth, cropHeight);
    dst[3] = cv::Point2f(0.0f, cropHeight);

    cv::Mat transform = cv::getPerspectiveTransform(srcPts, dst);
    cv::Mat partImage;
    cv::warpPerspective(cropped, partImage, transform, cv::Size(cropWidth, cropHeight), cv::INTER_CUBIC, cv::BORDER_REPLICATE);

    if (partImage.rows >= partImage.cols * 1.5f) {
        cv::Mat rotated;
        cv::transpose(partImage, rotated);
        cv::flip(rotated, rotated, 0);
        return rotated;
    }
    return partImage;
}

cv::Mat FitToSize(const cv::Mat& src, int dstWidth, int dstHeight) {
    cv::Mat resized;
    const float scale = static_cast<float>(dstHeight) / static_cast<float>(src.rows);
    const int scaledWidth = static_cast<int>(static_cast<float>(src.cols) * scale);
    cv::resize(src, resized, cv::Size(scaledWidth, dstHeight));
    cv::Mat fitted(dstHeight, dstWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    if (scaledWidth < dstWidth) {
        resized.copyTo(fitted(cv::Rect(0, 0, resized.cols, resized.rows)));
    } else {
        resized(cv::Rect(0, 0, dstWidth, dstHeight)).copyTo(fitted);
    }
    return fitted;
}

std::vector<cv::Point2f> GetMinBoxes(const cv::RotatedRect& boxRect, float& maxSideLen) {
    maxSideLen = std::min(boxRect.size.width, boxRect.size.height);

    std::vector<cv::Point2f> boxPoints = GetBox(boxRect);
    std::sort(boxPoints.begin(), boxPoints.end(), ComparePointX);

    int index1 = 0;
    int index2 = 0;
    int index3 = 0;
    int index4 = 0;
    if (boxPoints[1].y > boxPoints[0].y) {
        index1 = 0;
        index4 = 1;
    } else {
        index1 = 1;
        index4 = 0;
    }
    if (boxPoints[3].y > boxPoints[2].y) {
        index2 = 2;
        index3 = 3;
    } else {
        index2 = 3;
        index3 = 2;
    }

    std::vector<cv::Point2f> minBox(4);
    minBox[0] = boxPoints[index1];
    minBox[1] = boxPoints[index2];
    minBox[2] = boxPoints[index3];
    minBox[3] = boxPoints[index4];
    return minBox;
}

float BoxScoreFast(const std::vector<cv::Point2f>& boxes, const cv::Mat& pred) {
    const int width = pred.cols;
    const int height = pred.rows;
    const float arrayX[4] = {boxes[0].x, boxes[1].x, boxes[2].x, boxes[3].x};
    const float arrayY[4] = {boxes[0].y, boxes[1].y, boxes[2].y, boxes[3].y};

    const int minX = ClampInt(static_cast<int>(std::floor(*std::min_element(arrayX, arrayX + 4))), 0, width - 1);
    const int maxX = ClampInt(static_cast<int>(std::ceil(*std::max_element(arrayX, arrayX + 4))), 0, width - 1);
    const int minY = ClampInt(static_cast<int>(std::floor(*std::min_element(arrayY, arrayY + 4))), 0, height - 1);
    const int maxY = ClampInt(static_cast<int>(std::ceil(*std::max_element(arrayY, arrayY + 4))), 0, height - 1);

    cv::Mat mask = cv::Mat::zeros(maxY - minY + 1, maxX - minX + 1, CV_8UC1);
    cv::Point box[4];
    for (int i = 0; i < 4; ++i) {
        box[i] = cv::Point(
            static_cast<int>(boxes[i].x) - minX,
            static_cast<int>(boxes[i].y) - minY);
    }
    const cv::Point* pts[1] = {box};
    const int npts[] = {4};
    cv::fillPoly(mask, pts, npts, 1, cv::Scalar(1));

    cv::Mat croppedImg;
    pred(cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1)).copyTo(croppedImg);
    return static_cast<float>(cv::mean(croppedImg, mask)[0]);
}

float BoxScoreSlow(const std::vector<cv::Point>& contour, const cv::Mat& pred) {
    if (contour.empty()) {
        return 0.0f;
    }

    const int width = pred.cols;
    const int height = pred.rows;

    int minX = width - 1;
    int maxX = 0;
    int minY = height - 1;
    int maxY = 0;
    for (std::size_t i = 0; i < contour.size(); ++i) {
        minX = std::min(minX, contour[i].x);
        maxX = std::max(maxX, contour[i].x);
        minY = std::min(minY, contour[i].y);
        maxY = std::max(maxY, contour[i].y);
    }

    minX = ClampInt(minX, 0, width - 1);
    maxX = ClampInt(maxX, 0, width - 1);
    minY = ClampInt(minY, 0, height - 1);
    maxY = ClampInt(maxY, 0, height - 1);

    cv::Mat mask = cv::Mat::zeros(maxY - minY + 1, maxX - minX + 1, CV_8UC1);
    std::vector<cv::Point> shifted = contour;
    for (std::size_t i = 0; i < shifted.size(); ++i) {
        shifted[i].x -= minX;
        shifted[i].y -= minY;
    }

    const cv::Point* pts[1] = {shifted.data()};
    const int npts[] = {static_cast<int>(shifted.size())};
    cv::fillPoly(mask, pts, npts, 1, cv::Scalar(1));

    cv::Mat croppedImg;
    pred(cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1)).copyTo(croppedImg);
    return static_cast<float>(cv::mean(croppedImg, mask)[0]);
}

cv::RotatedRect Unclip(const std::vector<cv::Point2f>& box, float unclipRatio) {
    const float distance = GetContourArea(box, unclipRatio);
    constexpr double kClipperScale = 1024.0;

    ClipperLib::ClipperOffset offset;
    ClipperLib::Path path;
    path << ClipperLib::IntPoint(
                static_cast<ClipperLib::cInt>(std::llround(box[0].x * kClipperScale)),
                static_cast<ClipperLib::cInt>(std::llround(box[0].y * kClipperScale)))
         << ClipperLib::IntPoint(
                static_cast<ClipperLib::cInt>(std::llround(box[1].x * kClipperScale)),
                static_cast<ClipperLib::cInt>(std::llround(box[1].y * kClipperScale)))
         << ClipperLib::IntPoint(
                static_cast<ClipperLib::cInt>(std::llround(box[2].x * kClipperScale)),
                static_cast<ClipperLib::cInt>(std::llround(box[2].y * kClipperScale)))
         << ClipperLib::IntPoint(
                static_cast<ClipperLib::cInt>(std::llround(box[3].x * kClipperScale)),
                static_cast<ClipperLib::cInt>(std::llround(box[3].y * kClipperScale)));

    offset.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

    ClipperLib::Paths solution;
    offset.Execute(solution, static_cast<double>(distance) * kClipperScale);

    std::vector<cv::Point2f> points;
    for (std::size_t j = 0; j < solution.size(); ++j) {
        for (std::size_t i = 0; i < solution[j].size(); ++i) {
            points.emplace_back(
                static_cast<float>(solution[j][i].X / kClipperScale),
                static_cast<float>(solution[j][i].Y / kClipperScale));
        }
    }

    if (points.empty()) {
        return cv::RotatedRect(cv::Point2f(0.0f, 0.0f), cv::Size2f(1.0f, 1.0f), 0.0f);
    }
    return cv::minAreaRect(points);
}

std::vector<float> SubtractMeanNormalize(
    const cv::Mat& src,
    const std::array<float, 3>& meanVals,
    const std::array<float, 3>& normVals) {
    
    const int channels = src.channels();
    if (channels != 3) {
        throw std::invalid_argument("Only 3-channel images are supported!");
    }
    
    const std::size_t imageSize = static_cast<std::size_t>(src.cols) * src.rows;
    const std::size_t totalSize = imageSize * channels;

    std::vector<float> tensorValues(totalSize);
    
    const float invNorm[3] = {
        1.0f / (255.0f * normVals[0]),
        1.0f / (255.0f * normVals[1]),
        1.0f / (255.0f * normVals[2])
    };
    
    const float meanScaled[3] = {
        meanVals[0] / normVals[0],
        meanVals[1] / normVals[1],
        meanVals[2] / normVals[2]
    };

    const unsigned char* srcData = src.data;
    float* ch0 = tensorValues.data();
    float* ch1 = ch0 + imageSize;
    float* ch2 = ch1 + imageSize;

    for (std::size_t i = 0; i < imageSize; ++i) {
        const std::size_t idx = i * channels;
        ch0[i] = static_cast<float>(srcData[idx]) * invNorm[0] - meanScaled[0];
        ch1[i] = static_cast<float>(srcData[idx + 1]) * invNorm[1] - meanScaled[1];
        ch2[i] = static_cast<float>(srcData[idx + 2]) * invNorm[2] - meanScaled[2];
    }

    return tensorValues;
}

std::vector<int> GetAngleIndexes(const std::vector<AnglePrediction>& angles) {
    std::vector<int> indexes;
    indexes.reserve(angles.size());
    for (const AnglePrediction& angle : angles) {
        indexes.push_back(angle.index);
    }
    return indexes;
}

std::vector<Ort::AllocatedStringPtr> GetInputNames(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t inputCount = session.GetInputCount();
    std::vector<Ort::AllocatedStringPtr> names;
    names.reserve(inputCount);
    for (std::size_t i = 0; i < inputCount; ++i) {
        names.push_back(session.GetInputNameAllocated(i, allocator));
    }
    return names;
}

std::vector<Ort::AllocatedStringPtr> GetOutputNames(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t outputCount = session.GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> names;
    names.reserve(outputCount);
    for (std::size_t i = 0; i < outputCount; ++i) {
        names.push_back(session.GetOutputNameAllocated(i, allocator));
    }
    return names;
}

cv::Mat DecodeImageBytes(const std::vector<unsigned char>& bytes) {
    if (bytes.empty()) {
        return cv::Mat();
    }

    int width = 0;
    int height = 0;
    int channels = 0;

    unsigned char* decoded = stbi_load_from_memory(
        bytes.data(),
        static_cast<int>(bytes.size()),
        &width,
        &height,
        &channels,
        3
    );

    if (decoded == nullptr || width <= 0 || height <= 0) {
        if (decoded != nullptr) {
            stbi_image_free(decoded);
        }
        return cv::Mat();
    }

    cv::Mat rgb(height, width, CV_8UC3, decoded);
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

    stbi_image_free(decoded);
    return bgr;
}

}  // namespace rapidocr