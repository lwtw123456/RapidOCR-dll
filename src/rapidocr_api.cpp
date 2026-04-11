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
#include <filesystem>
#include <cctype>

#ifdef _WIN32
#include <Windows.h>
#endif

#include <opencv2/core.hpp>

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
    bool onlyText;

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
          mergeCodeLines(false),
          onlyText(false) {}
};

struct EngineHolder {
    std::mutex mutex;
    OcrEngine engine;
};

thread_local std::string g_lastJson;

static void AppendEscapedJson(std::string& out, const std::string& s) {
    out.push_back('"');
    for (std::size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        switch (c) {
            case '"':  out.append("\\\""); break;
            case '\\': out.append("\\\\"); break;
            case '\b': out.append("\\b");  break;
            case '\f': out.append("\\f");  break;
            case '\n': out.append("\\n");  break;
            case '\r': out.append("\\r");  break;
            case '\t': out.append("\\t");  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out.append(buf);
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    out.push_back('"');
}

static std::string MakeResultString(int code, const std::string& message) {
    std::string out;
    out.append("{\"code\":");
    out.append(std::to_string(code));
    out.append(",\"message\":");
    AppendEscapedJson(out, message);
    out.append(",\"data\":[]}");
    return out;
}

const char* ReturnFallbackJson(int code, const char* message) noexcept {
    try {
        g_lastJson = MakeResultString(code, message == NULL ? "" : message);
    } catch (...) {
        g_lastJson = std::string("{\"code\":") + std::to_string(code) +
                     ",\"message\":\"internal json serialization error\",\"data\":[]}";
    }
    return g_lastJson.c_str();
}

const char* ReturnJson(const std::string& jsonStr) noexcept {
    try {
        g_lastJson = jsonStr;
        return g_lastJson.c_str();
    } catch (...) {
        return ReturnFallbackJson(RC_INTERNAL_ERROR, "internal json serialization error");
    }
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

static std::string BuildSuccessString(const OcrResult& result) {
    std::string out;
    out.append("{\"code\":100,\"message\":\"ok\",\"data\":[");
    bool first = true;
    for (std::size_t i = 0; i < result.textBlocks.size(); ++i) {
        const TextBlock& block = result.textBlocks[i];
        if (block.text.empty()) { continue; }
        if (!first) out.push_back(',');
        first = false;

        out.append("{\"box\":[");
        for (std::size_t p = 0; p < block.boxPoints.size(); ++p) {
            if (p > 0) out.push_back(',');
            out.push_back('[');
            out.append(std::to_string(block.boxPoints[p].x));
            out.push_back(',');
            out.append(std::to_string(block.boxPoints[p].y));
            out.push_back(']');
        }
        out.append("],");

        char scoreBuf[32];
        std::snprintf(scoreBuf, sizeof(scoreBuf), "%.5f", CalcScore(block));
        out.append("\"score\":");
        out.append(scoreBuf);
        out.append(",\"text\":");
        AppendEscapedJson(out, block.text);
        out.push_back('}');
    }
    out.append("]}");
    return out;
}

static std::string BuildOnlyTextString(const OcrResult& result) {
    std::string out;
    bool first = true;
    for (std::size_t i = 0; i < result.textBlocks.size(); ++i) {
        const TextBlock& block = result.textBlocks[i];
        if (block.text.empty()) {
            continue;
        }
        if (!first) {
            out.push_back('\n');
        }
        out.append(block.text);
        first = false;
    }
    return out;
}

const char* ReturnExceptionJson(int code, const std::exception& ex) noexcept {
    return ReturnFallbackJson(code, ex.what());
}

const char* ReturnUnknownJson(int code, const char* message) noexcept {
    return ReturnFallbackJson(code, message);
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
    namespace fs = std::filesystem;
    return (fs::path(dir) / fileName).string();
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

std::string ToLowerCopy(const std::string& s) {
    std::string out = s;
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(out[i])));
    }
    return out;
}

bool ContainsToken(const std::string& text, const std::string& token) {
    return text.find(token) != std::string::npos;
}

std::string FindModelByPattern(const std::string& modelDir, const std::string& keyword) {
    namespace fs = std::filesystem;

    fs::path dir(modelDir);
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        throw std::runtime_error("model directory does not exist: " + modelDir);
    }

    std::vector<std::string> matches;

    for (fs::directory_iterator it(dir); it != fs::directory_iterator(); ++it) {
        if (!it->is_regular_file()) {
            continue;
        }

        const fs::path path = it->path();
        const std::string ext = ToLowerCopy(path.extension().string());
        if (ext != ".onnx") {
            continue;
        }

        const std::string name = ToLowerCopy(path.filename().string());
        if (ContainsToken(name, keyword)) {
            matches.push_back(path.string());
        }
    }

    if (matches.empty()) {
        throw std::runtime_error("no model matched *" + keyword + "*.onnx in: " + modelDir);
    }

    std::sort(matches.begin(), matches.end());

    if (matches.size() > 1) {
        throw std::runtime_error(
            "multiple models matched *" + keyword + "*.onnx in: " + modelDir +
            ", please keep only one matching file");
    }

    return matches[0];
}


static bool ExtractJsonString(const std::string& json, const char* key, std::string& outValue) {
    std::string needle = std::string("\"") + key + "\"";
    std::size_t pos = json.find(needle);
    if (pos == std::string::npos) return false;
    
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;
    
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return false;
    
    pos++;
    std::string value;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            pos++;
        }
        value.push_back(json[pos++]);
    }
    outValue = value;
    return true;
}

static bool ExtractJsonNumber(const std::string& json, const char* key, std::string& outValue) {
    const std::string needle = std::string("\"") + key + "\"";
    std::size_t pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos += needle.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    if (pos >= json.size() || json[pos] != ':') return false;
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    std::string value;
    while (pos < json.size() && (std::isdigit(static_cast<unsigned char>(json[pos]))
           || json[pos] == '.' || json[pos] == '-' || json[pos] == '+')) {
        value.push_back(json[pos++]);
    }
    if (value.empty()) return false;
    outValue = value;
    return true;
}

static bool ExtractJsonBool(const std::string& json, const char* key, bool defaultValue) {
    const std::string needle = std::string("\"") + key + "\"";
    std::size_t pos = json.find(needle);
    if (pos == std::string::npos) return defaultValue;
    pos += needle.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    if (pos >= json.size() || json[pos] != ':') return defaultValue;
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    if (pos + 3 < json.size() && json.substr(pos, 4) == "true")  return true;
    if (pos + 4 < json.size() && json.substr(pos, 5) == "false") return false;
    if (pos < json.size() && json[pos] == '1') return true;
    if (pos < json.size() && json[pos] == '0') return false;
    return defaultValue;
}

static int ParseInt(const std::string& s, int defaultValue) {
    if (s.empty()) return defaultValue;
    try { return std::stoi(s); } catch (...) { return defaultValue; }
}

static float ParseFloat(const std::string& s, float defaultValue) {
    if (s.empty()) return defaultValue;
    try { return std::stof(s); } catch (...) { return defaultValue; }
}

ApiOptions ParseOptions(const char* optionsJson) {
    ApiOptions options;
    if (optionsJson == NULL || optionsJson[0] == '\0') {
        options.modelDir = NormalizeModelDir("");
        return options;
    }
    const std::string json(optionsJson);
    if (json.find('{') == std::string::npos) {
        options.modelDir = NormalizeModelDir("");
        return options;
    }

    std::string sv;
    options.modelDir       = NormalizeModelDir(ExtractJsonString(json, "model_dir", sv) ? sv : "");
    options.useCls         = ExtractJsonBool(json, "use_cls",         options.useCls);
    options.useDilation    = ExtractJsonBool(json, "use_dilation",    options.useDilation);
    options.mergeCodeLines = ExtractJsonBool(json, "merge_code_lines",options.mergeCodeLines);
	options.onlyText       = ExtractJsonBool(json, "only_text",        options.onlyText);

    if (ExtractJsonNumber(json, "max_side_len",   sv)) options.maxSideLen    = std::max(1, ParseInt(sv, options.maxSideLen));
    if (ExtractJsonNumber(json, "min_side_len",   sv)) options.minSideLen    = std::max(1, ParseInt(sv, options.minSideLen));
    if (ExtractJsonNumber(json, "limit_side_len", sv)) options.limitSideLen  = ParseFloat(sv, options.limitSideLen);
    if (ExtractJsonNumber(json, "thresh",         sv)) options.thresh        = ParseFloat(sv, options.thresh);
    if (ExtractJsonNumber(json, "box_thresh",     sv)) options.boxThresh     = ParseFloat(sv, options.boxThresh);
    if (ExtractJsonNumber(json, "max_candidates", sv)) options.maxCandidates = std::max(1, ParseInt(sv, options.maxCandidates));
    if (ExtractJsonNumber(json, "unclip_ratio",   sv)) options.unclipRatio   = ParseFloat(sv, options.unclipRatio);
    if (ExtractJsonString(json, "limit_type",     sv)) options.limitType     = sv;
    if (ExtractJsonString(json, "score_mode",     sv)) options.scoreMode     = sv;

    if (options.limitSideLen <= 0.0f)  options.limitSideLen = 736.0f;
    if (options.limitType != "min" && options.limitType != "max") options.limitType = "min";
    if (options.thresh    < 0.0f || options.thresh    > 1.0f) options.thresh    = 0.3f;
    if (options.boxThresh < 0.0f || options.boxThresh > 1.0f) options.boxThresh = 0.5f;
    if (options.unclipRatio <= 0.0f)   options.unclipRatio = 1.6f;
    if (options.scoreMode != "fast" && options.scoreMode != "slow") options.scoreMode = "fast";
    if (options.maxSideLen < options.minSideLen) options.maxSideLen = options.minSideLen;
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
		modelPaths.detectorPath = FindModelByPattern(options.modelDir, "_det");
		modelPaths.classifierPath = FindModelByPattern(options.modelDir, "_cls");
		modelPaths.recognizerPath = FindModelByPattern(options.modelDir, "_rec");

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
        return ReturnJson(MakeResultString(RC_BAD_REQUEST, "empty image bytes"));
    }

    const ApiOptions options = ParseOptions(optionsJson);
    cv::Mat image = DecodeImageBytes(bytes);
    if (image.empty()) {
        return ReturnJson(MakeResultString(RC_BAD_REQUEST, "failed to decode image"));
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

    if (options.onlyText) {
		return ReturnJson(BuildOnlyTextString(result));
	}
	return ReturnJson(BuildSuccessString(result));
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
            return ReturnJson(MakeResultString(RC_BAD_REQUEST, "imagePath is empty"));
        }
        const std::vector<unsigned char> bytes = ReadFileBytesW(imagePath);
        if (bytes.empty()) {
            return ReturnJson(MakeResultString(RC_NOT_FOUND, "failed to read image file"));
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
            return ReturnJson(MakeResultString(RC_BAD_REQUEST, "imageBytes is empty"));
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

#ifdef _WIN32
BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID reserved) {
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(module);
    }
    return TRUE;
}
#endif
