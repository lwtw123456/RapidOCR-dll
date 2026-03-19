#ifndef RAPIDOCR_OCR_ENGINE_H_
#define RAPIDOCR_OCR_ENGINE_H_

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "ocr_types.h"
#include "classifier.h"
#include "detector.h"
#include "recognizer.h"

namespace rapidocr {

class OcrEngine {
public:
    OcrEngine();
    ~OcrEngine() = default;

    OcrEngine(const OcrEngine&) = delete;
    OcrEngine& operator=(const OcrEngine&) = delete;
    OcrEngine(OcrEngine&&) noexcept = default;
    OcrEngine& operator=(OcrEngine&&) noexcept = default;

    void InitializeModels(const OcrModelPaths& modelPaths);
    OcrResult Detect(const cv::Mat& image, const OcrRunOptions& options);

private:
    std::vector<cv::Mat> ExtractPartImages(
        const cv::Mat& src,
        const std::vector<TextBox>& textBoxes) const;

    OcrResult DetectImpl(
        const cv::Mat& src,
        const OcrRunOptions& options);

    Detector detector_;
    Classifier classifier_;
    Recognizer recognizer_;
};

}  // namespace rapidocr

#endif  // RAPIDOCR_OCR_ENGINE_H_