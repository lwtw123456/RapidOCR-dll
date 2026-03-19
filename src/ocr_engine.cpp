#include "ocr_engine.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include <opencv2/imgproc.hpp>

#include "ocr_common.h"

namespace rapidocr {

OcrEngine::OcrEngine() = default;

void OcrEngine::InitializeModels(const OcrModelPaths& modelPaths) {
    detector_.Initialize(modelPaths.detectorPath);
    classifier_.Initialize(modelPaths.classifierPath);
    recognizer_.Initialize(modelPaths.recognizerPath);
}

std::vector<cv::Mat> OcrEngine::ExtractPartImages(
    const cv::Mat& src,
    const std::vector<TextBox>& textBoxes) const {
    std::vector<cv::Mat> partImages;
    partImages.reserve(textBoxes.size());
    for (const TextBox& textBox : textBoxes) {
        partImages.push_back(GetRotateCropImage(src, textBox.boxPoints));
    }
    return partImages;
}

OcrResult OcrEngine::Detect(const cv::Mat& image, const OcrRunOptions& options) {
    if (image.empty()) {
        return {};
    }

    float resizeScale = 1.0f;
    cv::Mat workingImage = ResizeBySideLimit(image, options.minSideLen, options.maxSideLen, resizeScale);
    if (workingImage.empty()) {
        return {};
    }

    OcrResult result = DetectImpl(workingImage, options);

    if (std::fabs(resizeScale - 1.0f) > 1e-6f && resizeScale > 0.0f) {
        for (std::size_t i = 0; i < result.textBlocks.size(); ++i) {
            for (std::size_t j = 0; j < result.textBlocks[i].boxPoints.size(); ++j) {
                result.textBlocks[i].boxPoints[j].x = static_cast<int>(std::round(
                    static_cast<float>(result.textBlocks[i].boxPoints[j].x) / resizeScale));
                result.textBlocks[i].boxPoints[j].y = static_cast<int>(std::round(
                    static_cast<float>(result.textBlocks[i].boxPoints[j].y) / resizeScale));
            }
        }
    }

    return result;
}

OcrResult OcrEngine::DetectImpl(
    const cv::Mat& src,
    const OcrRunOptions& options) {

    std::vector<TextBox> textBoxes = detector_.Detect(src, options);

	std::vector<cv::Mat> partImages = ExtractPartImages(src, textBoxes);

	std::vector<AnglePrediction> angles(partImages.size(), AnglePrediction{-1, 0.0f});
	if (options.useCls) {
		angles = classifier_.Predict(partImages);

		for (std::size_t i = 0; i < partImages.size() && i < angles.size(); ++i) {
			if (classifier_.ShouldRotate180(angles[i])) {
				partImages[i] = Rotate180(partImages[i]);
			}
		}
	}

	std::vector<TextLine> textLines = recognizer_.Recognize(partImages);

    const std::size_t blockCount = std::min(textBoxes.size(), std::min(angles.size(), textLines.size()));
    std::vector<TextBlock> textBlocks;
    textBlocks.reserve(blockCount);

    for (std::size_t i = 0; i < blockCount; ++i) {
        textBlocks.push_back(TextBlock{
            textBoxes[i].boxPoints,
            textBoxes[i].score,
            angles[i].index,
            angles[i].score,
            textLines[i].text,
            textLines[i].charScores});
    }

    std::ostringstream combined;
    for (const TextBlock& block : textBlocks) {
        combined << block.text << '\n';
    }

    OcrResult result;
    result.textBlocks = std::move(textBlocks);
    result.combinedText = combined.str();
    return result;
}

}  // namespace rapidocr
