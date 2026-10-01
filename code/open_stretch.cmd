@echo off
setlocal
set "BLENDER=%~dp0..\..\build_windows_x64_Release\bin\blender.exe"
set "ASSET=%~dp0..\assets\plain-knit-stretch-v2\swatch.blend"
if not exist "%BLENDER%" (
    echo Blender executable not found. Edit BLENDER in this launcher.
    pause
    exit /b 1
)
if exist "%ASSET%" (
    "%BLENDER%" "%ASSET%" --python "%~dp0install_addon.py"
) else (
    "%BLENDER%" --python "%~dp0install_addon.py"
)
endlocal
