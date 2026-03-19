@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
set "ROOT_DIR=%cd%"
set "BUILD_DIR=%ROOT_DIR%\build"
set "CONFIG=Release"
set "GENERATOR=NMake Makefiles"
set "TARGET_DLL_NAME=RapidOCR.dll"

set "STB_DIR=%ROOT_DIR%\third_party\stb"
set "STB_IMAGE_H=%STB_DIR%\stb_image.h"
set "STB_IMAGE_WRITE_H=%STB_DIR%\stb_image_write.h"

set "ORT_DIR=%ROOT_DIR%\third_party\onnxruntime-static"
set "ORT_INCLUDE_DIR=%ORT_DIR%\include"
set "ORT_LIB_DIR=%ORT_DIR%\lib"
set "ORT_HEADER=%ORT_INCLUDE_DIR%\onnxruntime\core\session\onnxruntime_cxx_api.h"

echo [INFO] Root: %ROOT_DIR%

if not exist "%ROOT_DIR%\CMakeLists.txt" (
    echo [ERROR] CMakeLists.txt not found
    exit /b 1
)

if not exist "%ORT_HEADER%" (
    echo [ERROR] %ORT_HEADER% not found
    exit /b 1
)

if not exist "%ORT_LIB_DIR%" (
    echo [ERROR] %ORT_LIB_DIR% not found
    exit /b 1
)

if not exist "%STB_IMAGE_H%" (
    echo [ERROR] %STB_IMAGE_H% not found
    echo [HINT] Place stb_image.h under third_party\stb
    exit /b 1
)

where cmake >nul 2>nul
if errorlevel 1 (
    echo [ERROR] cmake not found in PATH
    exit /b 1
)

call :ensure_msvc
if errorlevel 1 exit /b 1

echo [INFO] cmake version:
cmake --version

if exist "%BUILD_DIR%" (
    echo [INFO] Removing old build directory...
    rmdir /s /q "%BUILD_DIR%"
    if exist "%BUILD_DIR%" (
        echo [ERROR] Failed to remove old build directory
        exit /b 1
    )
)

mkdir "%BUILD_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to create build directory
    exit /b 1
)

pushd "%BUILD_DIR%"

echo [INFO] Configuring...
cmake "%ROOT_DIR%" ^
    -G "%GENERATOR%" ^
    -DCMAKE_BUILD_TYPE=%CONFIG% ^
    -DRAPIDOCR_BUILD_SHARED=ON ^
    -DRAPIDOCR_USE_STATIC_RUNTIME=ON
if errorlevel 1 (
    popd
    echo [ERROR] CMake configure failed
    exit /b 1
)

echo [INFO] Building %CONFIG%...
cmake --build . --config %CONFIG%
if errorlevel 1 (
    popd
    echo [ERROR] Build failed
    exit /b 1
)

set "OUT1=%BUILD_DIR%\%TARGET_DLL_NAME%"
set "OUT2=%BUILD_DIR%\%CONFIG%\%TARGET_DLL_NAME%"
set "FOUND_DLL="

if exist "%OUT1%" set "FOUND_DLL=%OUT1%"
if not defined FOUND_DLL if exist "%OUT2%" set "FOUND_DLL=%OUT2%"

popd

if defined FOUND_DLL (
    echo [OK] Build succeeded
    echo [OK] DLL: %FOUND_DLL%

    where upx >nul 2>nul
    if errorlevel 1 (
        echo [INFO] UPX not found, skip compression
    ) else (
        echo [INFO] UPX found, compressing...
        upx -9 "%FOUND_DLL%"
        if errorlevel 1 (
            echo [WARN] UPX compression failed
        ) else (
            echo [OK] UPX compression done
        )
    )

    exit /b 0
)

echo [ERROR] Build completed but %TARGET_DLL_NAME% was not found
echo [INFO] Checked:
echo        %OUT1%
echo        %OUT2%
exit /b 1


:ensure_msvc
where cl >nul 2>nul
if not errorlevel 1 goto :msvc_ready

echo [INFO] cl.exe not found, trying to load Visual Studio environment...

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [ERROR] cl.exe not found and vswhere.exe not found
    echo [HINT] Please install Visual Studio Build Tools or open a Developer Command Prompt
    exit /b 1
)

set "VSINSTALL="
for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VSINSTALL=%%I"
)

if not defined VSINSTALL (
    echo [ERROR] Visual Studio with C++ tools was not found
    exit /b 1
)

if exist "%VSINSTALL%\Common7\Tools\VsDevCmd.bat" (
    call "%VSINSTALL%\Common7\Tools\VsDevCmd.bat" -host_arch=x64 -arch=x64
) else (
    echo [ERROR] VsDevCmd.bat not found under %VSINSTALL%
    exit /b 1
)

where cl >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Failed to initialize MSVC environment
    exit /b 1
)

:msvc_ready
where nmake >nul 2>nul
if errorlevel 1 (
    echo [ERROR] nmake.exe not found after MSVC environment setup
    exit /b 1
)

echo [INFO] MSVC:
cl 2>nul | findstr /i "Version"
exit /b 0