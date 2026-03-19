#ifndef RAPIDOCR_RECOGNIZER_H_
#define RAPIDOCR_RECOGNIZER_H_

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "ocr_types.h"

namespace rapidocr {

class Recognizer {
public:
    Recognizer();
    ~Recognizer() = default;

    Recognizer(const Recognizer&) = delete;
    Recognizer& operator=(const Recognizer&) = delete;
    Recognizer(Recognizer&&) noexcept = default;
    Recognizer& operator=(Recognizer&&) noexcept = default;

    void Initialize(const std::string& modelPath);

    std::vector<TextLine> Recognize(
        const std::vector<cv::Mat>& partImages) const;

private:
    struct RatioIndex {
        int index;
        float ratio;
    };

    void ConfigureSessionOptions();
    void ValidateReady() const;

    std::vector<std::string> LoadKeysFromModel() const;
    static std::vector<std::string> SplitCharacterList(const std::string& raw);

    std::vector<float> ResizeNormImg(
        const cv::Mat& img,
        float maxWhRatio) const;

    TextLine DecodeTextLine(
        const float* outputData,
        std::size_t steps,
        std::size_t classes) const;

    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<Ort::AllocatedStringPtr> inputNames_;
    std::vector<Ort::AllocatedStringPtr> outputNames_;

    int recBatchNum_ = 6;
    int imgChannels_ = 3;
    int imgHeight_ = 48;
    int imgWidth_ = 320;

    std::vector<std::string> keys_;
};

}  // namespace rapidocr

#endif  // RAPIDOCR_RECOGNIZER_H_