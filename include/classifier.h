#ifndef RAPIDOCR_CLASSIFIER_H_
#define RAPIDOCR_CLASSIFIER_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "ocr_types.h"

namespace rapidocr {

class Classifier {
public:
    Classifier();
    ~Classifier() = default;

    Classifier(const Classifier&) = delete;
    Classifier& operator=(const Classifier&) = delete;
    Classifier(Classifier&&) noexcept = default;
    Classifier& operator=(Classifier&&) noexcept = default;

    void Initialize(const std::string& modelPath);
	std::vector<AnglePrediction> Predict(
		const std::vector<cv::Mat>& partImages) const;

    bool ShouldRotate180(const AnglePrediction& prediction) const;

private:
    void ConfigureSessionOptions();
    void ValidateReady() const;
    cv::Mat ResizeNormImg(const cv::Mat& img) const;
    std::vector<AnglePrediction> PredictBatch(const std::vector<cv::Mat>& batchImages) const;

    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<Ort::AllocatedStringPtr> inputNames_;
    std::vector<Ort::AllocatedStringPtr> outputNames_;
    std::array<float, 3> meanValues_;
    std::array<float, 3> normValues_;
    std::array<std::string, 2> labelList_;
    int dstWidth_ = 192;
    int dstHeight_ = 48;
    int clsBatchNum_ = 6;
    float clsThresh_ = 0.9f;
};

}  // namespace rapidocr

#endif  // RAPIDOCR_CLASSIFIER_H_
