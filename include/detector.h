#ifndef RAPIDOCR_DETECTOR_H_
#define RAPIDOCR_DETECTOR_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "ocr_types.h"

namespace rapidocr {

class Detector {
public:
    Detector();
    ~Detector() = default;

    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;
    Detector(Detector&&) noexcept = default;
    Detector& operator=(Detector&&) noexcept = default;

    void Initialize(const std::string& modelPath);

    std::vector<TextBox> Detect(const cv::Mat& src, const OcrRunOptions& options) const;

private:
    void ConfigureSessionOptions();
    void ValidateReady() const;

    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<Ort::AllocatedStringPtr> inputNames_;
    std::vector<Ort::AllocatedStringPtr> outputNames_;
    std::array<float, 3> meanValues_;
    std::array<float, 3> normValues_;
};

}  // namespace rapidocr

#endif  // RAPIDOCR_DETECTOR_H_
