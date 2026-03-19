#include "rapidocr_api.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif

#include <opencv2/core.hpp>

#include "nlohmann_json.hpp"
#include "ocr_common.h"
#include "ocr_engine.h"
#include "ocr_types.h"

namespace {
using rapidocr::DecodeImageBytes;
using rapidocr::OcrEngine;
using rapidocr::OcrModelPaths;
using rapidocr::OcrResult;
using rapidocr::OcrRunOptions;
using rapidocr::RoundHalfToEven;
using rapidocr::TextBlock;

enum ResultCode {
    RC_OK = 100,
    RC_BAD_REQUEST = 400,
    RC_NOT_FOUND = 404,
    RC_INIT_FAILED = 500,
    RC_INTERNAL_ERROR = 501
};

struct ApiOptions {
    std::string modelDir;
    bool useCls;
    int maxSideLen;
    int minSideLen;
    float limitSideLen;
    std::string limitType;
    float thresh;
    float boxThresh;
    int maxCandidates;
    float unclipRatio;
    bool useDilation;
    std::string scoreMode;
    bool mergeCodeLines;

    ApiOptions()
        : modelDir(),
          useCls(true),
          maxSideLen(2000),
          minSideLen(30),
          limitSideLen(736.0f),
          limitType("min"),
          thresh(0.3f),
          boxThresh(0.5f),
          maxCandidates(1000),
          unclipRatio(1.6f),
          useDilation(true),
          scoreMode("fast"),
          mergeCodeLines(false) {}
};

struct EngineHolder {
    std::mutex mutex;
    OcrEngine engine;
};

thread_local std::string g_lastJson;
thread_local std::string g_lastLineText;

nlohmann::json MakeResultJson(int code, const std::string& message) {
    nlohmann::json json;
    json["code"] = code;
    json["message"] = message;
    json["data"] = nlohmann::json::array();
    return json;
}

const char* ReturnFallbackJson(int code, const char* message) noexcept {
    try {
        g_lastJson = MakeResultJson(code, message == NULL ? "" : message).dump();
    } catch (...) {
        g_lastJson = std::string("{\"code\":") + std::to_string(code) +
                     ",\"message\":\"internal json serialization error\",\"data\":[]}";
    }
    return g_lastJson.c_str();
}

const char* ReturnJson(const nlohmann::json& json) noexcept {
    try {
        g_lastJson = json.dump();
        return g_lastJson.c_str();
    } catch (...) {
        return ReturnFallbackJson(RC_INTERNAL_ERROR, "internal json serialization error");
    }
}

const char* ReturnExceptionJson(int code, const std::exception& ex) noexcept {
    return ReturnFallbackJson(code, ex.what());
}

const char* ReturnUnknownJson(int code, const char* message) noexcept {
    return ReturnFallbackJson(code, message);
}

const unsigned char* ReturnLineTextBytes(const std::string& text, int* outTextLength) noexcept {
    try {
        g_lastLineText = text;
        if (outTextLength != NULL) {
            *outTextLength = static_cast<int>(g_lastLineText.size());
        }
        return reinterpret_cast<const unsigned char*>(g_lastLineText.data());
    } catch (...) {
        g_lastLineText.clear();
        if (outTextLength != NULL) {
            *outTextLength = 0;
        }
        return reinterpret_cast<const unsigned char*>(g_lastLineText.data());
    }
}

const unsigned char* ReturnEmptyLineText(int* outTextLength) noexcept {
    g_lastLineText.clear();
    if (outTextLength != NULL) {
        *outTextLength = 0;
    }
    return reinterpret_cast<const unsigned char*>(g_lastLineText.data());
}

bool JsonGetBoolFlexible(const nlohmann::json& json, const char* key, bool defaultValue) {
    if (!json.contains(key)) {
        return defaultValue;
    }
    const nlohmann::json& value = json[key];
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    if (value.is_number_integer()) {
        return value.get<int>() != 0;
    }
    return defaultValue;
}

int JsonGetInt(const nlohmann::json& json, const char* key, int defaultValue) {
    if (!json.contains(key) || !json[key].is_number_integer()) {
        return defaultValue;
    }
    return json[key].get<int>();
}

float JsonGetFloat(const nlohmann::json& json, const char* key, float defaultValue) {
    if (!json.contains(key)) {
        return defaultValue;
    }
    const nlohmann::json& value = json[key];
    if (!(value.is_number_float() || value.is_number_integer())) {
        return defaultValue;
    }
    return value.get<float>();
}

std::string JsonGetString(const nlohmann::json& json, const char* key, const std::string& defaultValue) {
    if (!json.contains(key) || !json[key].is_string()) {
        return defaultValue;
    }
    return json[key].get<std::string>();
}

#ifdef _WIN32
std::wstring GetModuleDirW() {
    HMODULE module = NULL;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&GetModuleDirW),
            &module)) {
        return L".";
    }

    wchar_t path[MAX_PATH] = {0};
    if (GetModuleFileNameW(module, path, MAX_PATH) == 0) {
        return L".";
    }

    std::wstring fullPath(path);
    const std::size_t pos = fullPath.find_last_of(L"\\/");
    return pos == std::wstring::npos ? L"." : fullPath.substr(0, pos);
}

std::string WideToUtf8(const std::wstring& wide) {
    if (wide.empty()) {
        return std::string();
    }
    const int length = WideCharToMultiByte(
        CP_UTF8, 0, wide.c_str(), static_cast<int>(wide.size()), NULL, 0, NULL, NULL);
    if (length <= 0) {
        return std::string();
    }
    std::string output(length, '\0');
    WideCharToMultiByte(
        CP_UTF8, 0, wide.c_str(), static_cast<int>(wide.size()), &output[0], length, NULL, NULL);
    return output;
}
#else
std::string WideToUtf8(const std::wstring& wide) {
    return std::string(wide.begin(), wide.end());
}
#endif

std::string JoinPath(const std::string& dir, const char* fileName) {
    if (dir.empty()) {
        return std::string(fileName);
    }
    const char last = dir[dir.size() - 1];
    if (last == '\\' || last == '/') {
        return dir + fileName;
    }
#ifdef _WIN32
    return dir + "\\" + fileName;
#else
    return dir + "/" + fileName;
#endif
}

std::string GetDefaultModelDirUtf8() {
#ifdef _WIN32
    std::wstring baseDir = GetModuleDirW();
    if (!baseDir.empty() && baseDir[baseDir.size() - 1] != L'\\' && baseDir[baseDir.size() - 1] != L'/') {
        baseDir += L"\\";
    }
    baseDir += L"models";
    return WideToUtf8(baseDir);
#else
    return "models";
#endif
}

bool IsAbsolutePathUtf8(const std::string& path) {
    if (path.size() >= 2 && path[1] == ':') {
        return true;
    }
    if (path.size() >= 1 && (path[0] == '\\' || path[0] == '/')) {
        return true;
    }
    return false;
}

std::string NormalizeModelDir(const std::string& modelDir) {
    if (modelDir.empty()) {
        return GetDefaultModelDirUtf8();
    }
    if (IsAbsolutePathUtf8(modelDir)) {
        return modelDir;
    }
#ifdef _WIN32
    std::string base = WideToUtf8(GetModuleDirW());
    return JoinPath(base, modelDir.c_str());
#else
    return modelDir;
#endif
}

ApiOptions ParseOptions(const char* optionsJson) {
    ApiOptions options;
    if (optionsJson == NULL || optionsJson[0] == '\0') {
        options.modelDir = NormalizeModelDir("");
        return options;
    }

    const nlohmann::json json = nlohmann::json::parse(optionsJson, NULL, false);
    if (json.is_discarded() || !json.is_object()) {
        options.modelDir = NormalizeModelDir("");
        return options;
    }

    options.modelDir = NormalizeModelDir(JsonGetString(json, "model_dir", ""));
    options.useCls = JsonGetBoolFlexible(json, "use_cls", options.useCls);
    options.maxSideLen = std::max(1, JsonGetInt(json, "max_side_len", options.maxSideLen));
    options.minSideLen = std::max(1, JsonGetInt(json, "min_side_len", options.minSideLen));
    options.limitSideLen = JsonGetFloat(json, "limit_side_len", options.limitSideLen);
    options.limitType = JsonGetString(json, "limit_type", options.limitType);
    options.thresh = JsonGetFloat(json, "thresh", options.thresh);
    options.boxThresh = JsonGetFloat(json, "box_thresh", options.boxThresh);
    options.maxCandidates = std::max(1, JsonGetInt(json, "max_candidates", options.maxCandidates));
    options.unclipRatio = JsonGetFloat(json, "unclip_ratio", options.unclipRatio);
    options.useDilation = JsonGetBoolFlexible(json, "use_dilation", options.useDilation);
    options.scoreMode = JsonGetString(json, "score_mode", options.scoreMode);
    options.mergeCodeLines = JsonGetBoolFlexible(json, "merge_code_lines", options.mergeCodeLines);

    if (options.limitSideLen <= 0.0f) {
        options.limitSideLen = 736.0f;
    }
    if (options.limitType != "min" && options.limitType != "max") {
        options.limitType = "min";
    }
    if (options.thresh < 0.0f || options.thresh > 1.0f) {
        options.thresh = 0.3f;
    }
    if (options.boxThresh < 0.0f || options.boxThresh > 1.0f) {
        options.boxThresh = 0.5f;
    }
    if (options.unclipRatio <= 0.0f) {
        options.unclipRatio = 1.6f;
    }
    if (options.scoreMode != "fast" && options.scoreMode != "slow") {
        options.scoreMode = "fast";
    }
    if (options.maxSideLen < options.minSideLen) {
        options.maxSideLen = options.minSideLen;
    }
    return options;
}

std::vector<unsigned char> ReadFileBytesW(const wchar_t* path) {
    std::vector<unsigned char> data;
#ifdef _WIN32
    if (path == NULL || path[0] == L'\0') {
        return data;
    }
    FILE* fp = _wfopen(path, L"rb");
#else
    if (path == NULL || path[0] == 0) {
        return data;
    }
    std::string narrow = WideToUtf8(std::wstring(path));
    FILE* fp = fopen(narrow.c_str(), "rb");
#endif
    if (fp == NULL) {
        return data;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return data;
    }
    const long size = ftell(fp);
    if (size <= 0) {
        fclose(fp);
        return data;
    }
    rewind(fp);
    data.resize(static_cast<std::size_t>(size));
    const std::size_t readSize = fread(&data[0], 1, static_cast<std::size_t>(size), fp);
    fclose(fp);
    if (readSize != static_cast<std::size_t>(size)) {
        data.clear();
    }
    return data;
}

double CalcScore(const TextBlock& block) {
    if (!block.charScores.empty()) {
        double total = 0.0;
        for (std::size_t i = 0; i < block.charScores.size(); ++i) {
            total += static_cast<double>(block.charScores[i]);
        }
        const double mean = total / static_cast<double>(block.charScores.size());
        return RoundHalfToEven(mean, 5);
    }
    if (block.boxScore > 0.0f) {
        return RoundHalfToEven(static_cast<double>(block.boxScore), 5);
    }
    return 0.0;
}

nlohmann::json BuildSimpleBlockJson(const TextBlock& block) {
    nlohmann::json item;
    item["box"] = nlohmann::json::array();
    for (std::size_t p = 0; p < block.boxPoints.size(); ++p) {
        item["box"].push_back({block.boxPoints[p].x, block.boxPoints[p].y});
    }
    item["score"] = CalcScore(block);
    item["text"] = block.text;
    return item;
}

nlohmann::json BuildSuccessJson(const OcrResult& result) {
    nlohmann::json json;
    json["code"] = RC_OK;
    json["message"] = "ok";
    json["data"] = nlohmann::json::array();

    for (std::size_t i = 0; i < result.textBlocks.size(); ++i) {
        const TextBlock& block = result.textBlocks[i];
        if (block.text.empty()) {
            continue;
        }
        json["data"].push_back(BuildSimpleBlockJson(block));
    }

    return json;
}

std::string JoinLineTextsFromJsonBytes(const char* jsonBytes, int jsonBytesLength) {
    if (jsonBytes == NULL || jsonBytesLength <= 0) {
        return std::string();
    }

    const std::string jsonText(jsonBytes, jsonBytes + static_cast<std::size_t>(jsonBytesLength));
    const nlohmann::json root = nlohmann::json::parse(jsonText, NULL, false);
    if (root.is_discarded() || !root.is_object()) {
        return std::string();
    }

    if (!root.contains("data") || !root["data"].is_array()) {
        return std::string();
    }

    std::string output;
    const nlohmann::json& data = root["data"];
    bool first = true;

    for (std::size_t i = 0; i < data.size(); ++i) {
        const nlohmann::json& item = data[i];
        if (!item.is_object()) {
            continue;
        }
        if (!item.contains("text") || !item["text"].is_string()) {
            continue;
        }

        const std::string text = item["text"].get<std::string>();
        if (text.empty()) {
            continue;
        }

        if (!first) {
            output.push_back('\n');
        }
        output.append(text);
        first = false;
    }

    return output;
}

class EngineManager {
public:
    static EngineManager& Instance() {
        static EngineManager instance;
        return instance;
    }

    std::shared_ptr<EngineHolder> GetOrCreate(const ApiOptions& options) {
        const std::string key = BuildKey(options);
        std::lock_guard<std::mutex> lock(mutex_);
        std::map<std::string, std::shared_ptr<EngineHolder> >::iterator it = engines_.find(key);
        if (it != engines_.end()) {
            return it->second;
        }

        std::shared_ptr<EngineHolder> holder(new EngineHolder());
        OcrModelPaths modelPaths;
        modelPaths.detectorPath = JoinPath(options.modelDir, "ch_PP-OCRv5_mobile_det.onnx");
        modelPaths.classifierPath = JoinPath(options.modelDir, "ch_ppocr_mobile_v2.0_cls_infer.onnx");
        modelPaths.recognizerPath = JoinPath(options.modelDir, "ch_PP-OCRv5_rec_mobile_infer.onnx");

        holder->engine.InitializeModels(modelPaths);
        engines_[key] = holder;
        return holder;
    }

private:
    static std::string BuildKey(const ApiOptions& options) {
        return options.modelDir;
    }

    std::mutex mutex_;
    std::map<std::string, std::shared_ptr<EngineHolder> > engines_;
};

const char* RunOcrBytesCore(const std::vector<unsigned char>& bytes, const char* optionsJson) {
    if (bytes.empty()) {
        return ReturnJson(MakeResultJson(RC_BAD_REQUEST, "empty image bytes"));
    }

    const ApiOptions options = ParseOptions(optionsJson);
    cv::Mat image = DecodeImageBytes(bytes);
    if (image.empty()) {
        return ReturnJson(MakeResultJson(RC_BAD_REQUEST, "failed to decode image"));
    }

    std::shared_ptr<EngineHolder> holder = EngineManager::Instance().GetOrCreate(options);

    OcrRunOptions runOptions;
    runOptions.useCls = options.useCls;
    runOptions.maxSideLen = options.maxSideLen;
    runOptions.minSideLen = options.minSideLen;
    runOptions.limitSideLen = options.limitSideLen;
    runOptions.limitType = options.limitType;
    runOptions.thresh = options.thresh;
    runOptions.boxThresh = options.boxThresh;
    runOptions.maxCandidates = options.maxCandidates;
    runOptions.unclipRatio = options.unclipRatio;
    runOptions.useDilation = options.useDilation;
    runOptions.scoreMode = options.scoreMode;
    runOptions.mergeCodeLines = options.mergeCodeLines;

    OcrResult result;
    {
        std::lock_guard<std::mutex> lock(holder->mutex);
        result = holder->engine.Detect(image, runOptions);
    }

    return ReturnJson(BuildSuccessJson(result));
}

const char* RunOcrBytesInternal(const std::vector<unsigned char>& bytes, const char* optionsJson) noexcept {
    try {
        return RunOcrBytesCore(bytes, optionsJson);
    } catch (const std::exception& ex) {
        return ReturnExceptionJson(RC_INIT_FAILED, ex);
    } catch (...) {
        return ReturnUnknownJson(RC_INIT_FAILED, "unknown initialization error");
    }
}

}  // namespace

extern "C" RAPIDOCR_API const char* RAPIDOCR_CALL RapidOcrFromPathW(
    const wchar_t* imagePath,
    const char* optionsJson) noexcept {
    try {
        if (imagePath == NULL || imagePath[0] == 0) {
            return ReturnJson(MakeResultJson(RC_BAD_REQUEST, "imagePath is empty"));
        }
        const std::vector<unsigned char> bytes = ReadFileBytesW(imagePath);
        if (bytes.empty()) {
            return ReturnJson(MakeResultJson(RC_NOT_FOUND, "failed to read image file"));
        }
        return RunOcrBytesInternal(bytes, optionsJson);
    } catch (const std::exception& ex) {
        return ReturnExceptionJson(RC_INTERNAL_ERROR, ex);
    } catch (...) {
        return ReturnUnknownJson(RC_INTERNAL_ERROR, "unknown error");
    }
}

extern "C" RAPIDOCR_API const char* RAPIDOCR_CALL RapidOcrFromBytes(
    const unsigned char* imageBytes,
    int imageBytesLength,
    const char* optionsJson) noexcept {
    try {
        if (imageBytes == NULL || imageBytesLength <= 0) {
            return ReturnJson(MakeResultJson(RC_BAD_REQUEST, "imageBytes is empty"));
        }
        const std::vector<unsigned char> bytes(
            imageBytes,
            imageBytes + static_cast<std::size_t>(imageBytesLength));
        return RunOcrBytesInternal(bytes, optionsJson);
    } catch (const std::exception& ex) {
        return ReturnExceptionJson(RC_INTERNAL_ERROR, ex);
    } catch (...) {
        return ReturnUnknownJson(RC_INTERNAL_ERROR, "unknown error");
    }
}

extern "C" RAPIDOCR_API const unsigned char* RAPIDOCR_CALL RapidOcrJsonGetLineText(
    const char* jsonBytes,
    int jsonBytesLength,
    int* outTextLength) noexcept {
    try {
        return ReturnLineTextBytes(
            JoinLineTextsFromJsonBytes(jsonBytes, jsonBytesLength),
            outTextLength);
    } catch (...) {
        return ReturnEmptyLineText(outTextLength);
    }
}

#ifdef _WIN32
BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID reserved) {
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(module);
    }
    return TRUE;
}
#endif