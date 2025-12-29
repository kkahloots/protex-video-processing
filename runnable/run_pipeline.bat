@echo off
REM Protex AI - Pipeline Runner (Windows)
REM Usage: run_pipeline.bat [mode] [video_path]

setlocal enabledelayedexpansion

REM Change to parent directory
cd /d "%~dp0.."

REM Configuration
set MODE=%1
if "%MODE%"=="" set MODE=balanced

set VIDEO_PATH=%2
if "%VIDEO_PATH%"=="" set VIDEO_PATH=data\timelapse_test.mp4

set BATCH_SIZE=16
set NUM_SAMPLES=20

echo ============================================================
echo  Protex AI - Computer Vision Pipeline
echo ============================================================
echo Mode: %MODE%
echo Video: %VIDEO_PATH%
echo ============================================================
echo.

REM Check if video exists
if not exist "%VIDEO_PATH%" (
    echo Error: Video file not found: %VIDEO_PATH%
    exit /b 1
)

REM Stage 1: Preprocessing
echo Stage 1: Preprocessing (Video - Frames)
python 01_data_preprocessing.py ^
    --video_path "%VIDEO_PATH%" ^
    --mode "%MODE%" ^
    --verbose

if errorlevel 1 (
    echo Error in Stage 1
    exit /b 1
)

REM Stage 2: Pre-tagging
echo.
echo Stage 2: Pre-tagging (Frames - COCO Detections)
python 02_data_pretagging.py ^
    --mode "%MODE%" ^
    --batch_size %BATCH_SIZE% ^
    --verbose

if errorlevel 1 (
    echo Error in Stage 2
    exit /b 1
)

REM Stage 3: Cleanup
echo.
echo Stage 3: Cleanup (COCO - Cleaned COCO)
python 03_pretag_cleanup.py ^
    --mode "%MODE%" ^
    --verbose

if errorlevel 1 (
    echo Error in Stage 3
    exit /b 1
)

REM Stage 4: Sample Generation
echo.
echo Stage 4: Sample Generation
python 04_generate_samples.py ^
    --num_samples %NUM_SAMPLES% ^
    --verbose

if errorlevel 1 (
    echo Error in Stage 4
    exit /b 1
)

REM Stage 5: Report Generation
echo.
echo Stage 5: Report Generation
python 05_generate_report.py ^
    --verbose

if errorlevel 1 (
    echo Error in Stage 5
    exit /b 1
)

REM Stage 6: Presentation Generation
echo.
echo Stage 6: Presentation Generation
python 06_generate_presentation.py protex_presentation.mp4 ^
    --verbose

if errorlevel 1 (
    echo Error in Stage 6
    exit /b 1
)

echo.
echo ============================================================
echo Stage 7: Annotated Video Generation
echo ============================================================
python 07_generate_annotated_video.py ^
    --output traceables\sample_annotated_video.mp4 ^
    --fps 2.0 ^
    --verbose

if errorlevel 1 (
    echo Error in Stage 7
    exit /b 1
)

echo.
echo ============================================================
echo Pipeline completed successfully!
echo ============================================================
echo.
echo Outputs:
echo   - Frames: traceables\frames\
echo   - COCO: traceables\pre_tags\pre_tags_cleaned.json
echo   - Samples: traceables\samples\
echo   - Report: traceables\report\
echo   - Presentation: traceables\protex_presentation.mp4
echo   - Annotated Video: traceables\sample_annotated_video.mp4
echo ============================================================

endlocal
