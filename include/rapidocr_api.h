#ifndef RAPIDOCR_API_H_
#define RAPIDOCR_API_H_

#ifdef _WIN32
#define RAPIDOCR_API __declspec(dllexport)
#define RAPIDOCR_CALL __stdcall
#else
#define RAPIDOCR_API
#define RAPIDOCR_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

RAPIDOCR_API const char* RAPIDOCR_CALL RapidOcrFromPathW(
    const wchar_t* imagePath,
    const char* optionsJson) noexcept;

RAPIDOCR_API const char* RAPIDOCR_CALL RapidOcrFromBytes(
    const unsigned char* imageBytes,
    int imageBytesLength,
    const char* optionsJson) noexcept;

RAPIDOCR_API const unsigned char* RAPIDOCR_CALL RapidOcrJsonGetLineText(
    const char* jsonBytes,
    int jsonBytesLength,
    int* outTextLength) noexcept;

#ifdef __cplusplus
}
#endif

#endif  // RAPIDOCR_API_H_