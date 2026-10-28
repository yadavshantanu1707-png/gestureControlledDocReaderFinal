@echo off
echo ============================================
echo   Gesture Document Reader - Build Script
echo ============================================
echo.

echo [1/2] Installing dependencies ...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

echo.
echo [2/2] Building executable with PyInstaller ...
pyinstaller --clean gesture_reader.spec
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PyInstaller build failed.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   BUILD COMPLETE!
echo   Executable: dist\GestureDocReader.exe
echo ============================================
pause
