@echo off
REM ============================================================================
REM pbAudioStats Windows Build Script
REM ============================================================================
REM Usage: build_win.bat [Release|Debug]
REM ============================================================================

setlocal enabledelayedexpansion

REM Default to Release build
set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release

echo ============================================
echo pbAudioStats Windows Build
echo Build Type: %BUILD_TYPE%
echo ============================================

REM Create build directory
set BUILD_DIR=build_win
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

REM Change to build directory
cd "%BUILD_DIR%"

REM Configure with CMake (Visual Studio 2022, x64)
echo.
echo [1/2] Configuring with CMake...
cmake -G "Visual Studio 17 2022" -A x64 ..
if errorlevel 1 (
    echo ERROR: CMake configuration failed!
    exit /b 1
)

REM Build
echo.
echo [2/2] Building %BUILD_TYPE% configuration...
cmake --build . --config %BUILD_TYPE%
if errorlevel 1 (
    echo ERROR: Build failed!
    exit /b 1
)

REM Display result
echo.
echo ============================================
echo Build completed successfully!
echo ============================================
echo Executable: %BUILD_DIR%\%BUILD_TYPE%\pbAudioStats.exe
echo Library:    %BUILD_DIR%\%BUILD_TYPE%\pbAudioStatsLib.lib

REM Return to original directory
cd ..

exit /b 0
