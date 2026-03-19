#ifndef RAPIDOCR_OCR_TYPES_H_
#define RAPIDOCR_OCR_TYPES_H_

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace rapidocr {

struct ScaleParam {
    int srcWidth = 0;
    int srcHeight = 0;
    int dstWidth = 0;
    int dstHeight = 0;
    float ratioWidth = 1.0f;
    float ratioHeight = 1.0f;
};

struct TextBox {
    std::vector<cv::Point> boxPoints;
    float score = 0.0f;
};

struct AnglePrediction {
    int index = -1;
    float score = 0.0f;
};

struct TextLine {
    std::string text;
    std::vector<float> charScores;
};

struct TextBlock {
    std::vector<cv::Point> boxPoints;
    float boxScore = 0.0f;
    int angleIndex = -1;
    float angleScore = 0.0f;
    std::string text;
    std::vector<float> charScores;
};

struct OcrResult {
    std::vector<TextBlock> textBlocks;
    std::string combinedText;
};

struct OcrModelPaths {
    std::string detectorPath;
    std::string classifierPath;
    std::string recognizerPath;
};

struct OcrRunOptions {
    bool useCls = true;
    int maxSideLen = 2000;
    int minSideLen = 30;
    float limitSideLen = 736.0f;
    std::string limitType = "min";
    float thresh = 0.3f;
    float boxThresh = 0.5f;
    int maxCandidates = 1000;
    float unclipRatio = 1.6f;
    bool useDilation = true;
    std::string scoreMode = "fast";
    bool mergeCodeLines = false;
};

}  // namespace rapidocr

#endif  // RAPIDOCR_OCR_TYPES_H_
