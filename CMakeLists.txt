cmake_minimum_required(VERSION 3.21)

cmake_policy(SET CMP0091 NEW)

project(rapidocr
    VERSION 1.0.0
    DESCRIPTION "RapidOCR DLL wrapper"
    LANGUAGES CXX)

if (NOT WIN32)
    message(FATAL_ERROR "This project currently supports Windows only.")
endif()

if (NOT MSVC)
    message(FATAL_ERROR "This project currently supports MSVC only.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RAPIDOCR_BUILD_SHARED "Build rapidocr as a shared library" ON)
option(RAPIDOCR_USE_STATIC_RUNTIME "Use /MT or /MTd runtime on MSVC" ON)

if (RAPIDOCR_USE_STATIC_RUNTIME)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL$<$<CONFIG:Debug>:Debug>")
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(RAPIDOCR_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(RAPIDOCR_INCLUDE_DIR ${RAPIDOCR_ROOT_DIR}/include)
set(RAPIDOCR_SRC_DIR ${RAPIDOCR_ROOT_DIR}/src)
set(RAPIDOCR_3P_DIR ${RAPIDOCR_ROOT_DIR}/third_party)

set(STB_INCLUDE_DIR ${RAPIDOCR_3P_DIR}/stb)

set(OPENCV_ROOT_DIR ${RAPIDOCR_ROOT_DIR}/third_party/opencv-static)
set(OPENCV_INCLUDE_DIR ${OPENCV_ROOT_DIR}/include)
set(OPENCV_LIB_DIR ${OPENCV_ROOT_DIR}/lib)

set(OPENCV_CORE_LIB ${OPENCV_LIB_DIR}/opencv_core4130.lib)
set(OPENCV_IMGPROC_LIB ${OPENCV_LIB_DIR}/opencv_imgproc4130.lib)
set(OPENCV_ZLIB_LIB ${OPENCV_LIB_DIR}/zlib.lib)

set(ONNXRUNTIME_ROOT_DIR ${RAPIDOCR_ROOT_DIR}/third_party/onnxruntime-static)
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT_DIR}/include)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT_DIR}/lib)

if (NOT EXISTS ${STB_INCLUDE_DIR}/stb_image.h)
    message(FATAL_ERROR "Missing stb_image.h: ${STB_INCLUDE_DIR}/stb_image.h")
endif()

if (NOT EXISTS ${OPENCV_INCLUDE_DIR}/opencv2/core.hpp)
    message(FATAL_ERROR "Missing OpenCV headers: ${OPENCV_INCLUDE_DIR}/opencv2/core.hpp")
endif()

if (NOT EXISTS ${OPENCV_INCLUDE_DIR}/opencv2/imgproc.hpp)
    message(FATAL_ERROR "Missing OpenCV headers: ${OPENCV_INCLUDE_DIR}/opencv2/imgproc.hpp")
endif()

if (NOT EXISTS ${OPENCV_CORE_LIB})
    message(FATAL_ERROR "Missing OpenCV core lib: ${OPENCV_CORE_LIB}")
endif()

if (NOT EXISTS ${OPENCV_IMGPROC_LIB})
    message(FATAL_ERROR "Missing OpenCV imgproc lib: ${OPENCV_IMGPROC_LIB}")
endif()

if (NOT EXISTS ${OPENCV_ZLIB_LIB})
    message(FATAL_ERROR "Missing OpenCV zlib lib: ${OPENCV_ZLIB_LIB}")
endif()

if (NOT EXISTS ${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime/core/session/onnxruntime_cxx_api.h)
    message(FATAL_ERROR "Missing OnnxRuntime headers: ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

if (NOT EXISTS ${ONNXRUNTIME_LIB_DIR})
    message(FATAL_ERROR "Missing OnnxRuntime lib dir: ${ONNXRUNTIME_LIB_DIR}")
endif()

file(GLOB ONNXRUNTIME_LIBS
    "${ONNXRUNTIME_LIB_DIR}/*.lib"
)

if (NOT ONNXRUNTIME_LIBS)
    message(FATAL_ERROR "No OnnxRuntime libs found in: ${ONNXRUNTIME_LIB_DIR}")
endif()

list(REMOVE_DUPLICATES ONNXRUNTIME_LIBS)

set(RAPIDOCR_PUBLIC_HEADERS
    ${RAPIDOCR_INCLUDE_DIR}/ocr_common.h
    ${RAPIDOCR_INCLUDE_DIR}/ocr_engine.h
    ${RAPIDOCR_INCLUDE_DIR}/ocr_types.h
    ${RAPIDOCR_INCLUDE_DIR}/rapidocr_api.h
    ${RAPIDOCR_INCLUDE_DIR}/classifier.h
    ${RAPIDOCR_INCLUDE_DIR}/detector.h
    ${RAPIDOCR_INCLUDE_DIR}/recognizer.h
)

set(RAPIDOCR_SOURCES
    ${RAPIDOCR_SRC_DIR}/ocr_common.cpp
    ${RAPIDOCR_SRC_DIR}/ocr_engine.cpp
    ${RAPIDOCR_SRC_DIR}/rapidocr_api.cpp
    ${RAPIDOCR_SRC_DIR}/classifier.cpp
    ${RAPIDOCR_SRC_DIR}/detector.cpp
    ${RAPIDOCR_SRC_DIR}/recognizer.cpp
    ${RAPIDOCR_3P_DIR}/clipper/clipper.cpp
)

set(RAPIDOCR_3P_HEADERS
    ${RAPIDOCR_3P_DIR}/clipper/clipper.hpp
    ${RAPIDOCR_3P_DIR}/json/nlohmann_json.hpp
    ${STB_INCLUDE_DIR}/stb_image.h
)

if (RAPIDOCR_BUILD_SHARED)
    add_library(rapidocr SHARED)
else()
    add_library(rapidocr STATIC)
endif()

target_sources(rapidocr PRIVATE ${RAPIDOCR_SOURCES})

target_include_directories(rapidocr
    PRIVATE
        ${RAPIDOCR_INCLUDE_DIR}
        ${RAPIDOCR_3P_DIR}/clipper
        ${RAPIDOCR_3P_DIR}/json
        ${STB_INCLUDE_DIR}
        ${OPENCV_INCLUDE_DIR}
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_compile_definitions(rapidocr
    PRIVATE
        UNICODE
        _UNICODE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
)

target_compile_options(rapidocr
    PRIVATE
        /EHsc
        /permissive-
        /W4
        /wd4100
        /wd4127
        /wd4244
        /wd4267
        /wd4324
        /wd4456
        /wd4458
        /wd4702
        /Gy
)

target_link_options(rapidocr
    PRIVATE
        $<$<CONFIG:Release>:/OPT:REF /OPT:ICF>
)

target_link_libraries(rapidocr
    PRIVATE
        ${ONNXRUNTIME_LIBS}
        ${OPENCV_CORE_LIB}
        ${OPENCV_IMGPROC_LIB}
        ${OPENCV_ZLIB_LIB}
)

set_target_properties(rapidocr PROPERTIES
    OUTPUT_NAME "RapidOCR"
    DEBUG_POSTFIX "d"
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
)

source_group(TREE ${RAPIDOCR_ROOT_DIR} FILES
    ${RAPIDOCR_PUBLIC_HEADERS}
    ${RAPIDOCR_SOURCES}
    ${RAPIDOCR_3P_HEADERS}
)

message(STATUS "Configured ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "Using static MSVC runtime: ${RAPIDOCR_USE_STATIC_RUNTIME}")
message(STATUS "Build shared library: ${RAPIDOCR_BUILD_SHARED}")
message(STATUS "Using stb_image for image codec support")
message(STATUS "OpenCV include dir: ${OPENCV_INCLUDE_DIR}")
message(STATUS "OpenCV libs: ${OPENCV_CORE_LIB};${OPENCV_IMGPROC_LIB};${OPENCV_ZLIB_LIB}")
message(STATUS "OnnxRuntime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "OnnxRuntime libs: ${ONNXRUNTIME_LIBS}")