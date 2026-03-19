#ifndef RAPIDOCR_OCR_COMMON_H_
#define RAPIDOCR_OCR_COMMON_H_

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "ocr_types.h"

namespace rapidocr {

bool FileExists(const std::string& path);
std::wstring Utf8ToWide(const std::string& value);

int ClampInt(int value, int minValue, int maxValue);
int RoundHalfToEven(float value);
double RoundHalfToEven(double value, int digits);

ScaleParam GetScaleParam(const cv::Mat& src, float scale);
ScaleParam GetScaleParam(const cv::Mat& src, int targetSize);
ScaleParam GetDetScaleParam(const cv::Mat& src, float limitSideLen, const std::string& limitType);
cv::Mat ResizeBySideLimit(const cv::Mat& src, int minSideLen, int maxSideLen, float& scale);

cv::Mat Rotate180(const cv::Mat& src);
cv::Mat GetRotateCropImage(const cv::Mat& src, const std::vector<cv::Point>& box);
cv::Mat FitToSize(const cv::Mat& src, int dstWidth, int dstHeight);

std::vector<cv::Point2f> GetMinBoxes(const cv::RotatedRect& boxRect, float& minSideLen);
float BoxScoreFast(const std::vector<cv::Point2f>& boxes, const cv::Mat& pred);
float BoxScoreSlow(const std::vector<cv::Point>& contour, const cv::Mat& pred);
cv::RotatedRect Unclip(const std::vector<cv::Point2f>& box, float unclipRatio);

std::vector<float> SubtractMeanNormalize(
    const cv::Mat& src,
    const std::array<float, 3>& meanVals,
    const std::array<float, 3>& normVals);

std::vector<int> GetAngleIndexes(const std::vector<AnglePrediction>& angles);

std::vector<Ort::AllocatedStringPtr> GetInputNames(Ort::Session& session);
std::vector<Ort::AllocatedStringPtr> GetOutputNames(Ort::Session& session);

cv::Mat DecodeImageBytes(const std::vector<unsigned char>& bytes);

}  // namespace rapidocr

#endif  // RAPIDOCR_OCR_COMMON_H_