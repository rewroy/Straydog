@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "COMMIT_MSG=Update"

git rev-parse --show-toplevel >nul 2>&1
if errorlevel 1 (
  echo This directory is not a git repository.
  exit /b 1
)

echo === pull ===
git pull
if errorlevel 1 (
  echo Pull failed. Nothing was built, staged, committed, or pushed.
  exit /b 1
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo vswhere.exe was not found. Install Visual Studio Build Tools with the C++ workload.
  exit /b 1
)

set "VSINSTALL="
for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%I"
if not defined VSINSTALL (
  echo Visual Studio C++ tools were not found.
  exit /b 1
)

call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCINSTALLDIR (
  echo Failed to initialize the x64 Native Tools environment.
  exit /b 1
)

if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links" set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"

where cmake >nul 2>&1
if errorlevel 1 (
  echo cmake was not found. Install it with: winget install --id Kitware.CMake -e
  exit /b 1
)
where ninja >nul 2>&1
if errorlevel 1 (
  echo ninja was not found. Install it with: winget install --id Ninja-build.Ninja -e
  exit /b 1
)
where cargo >nul 2>&1
if errorlevel 1 (
  echo cargo was not found.
  exit /b 1
)

echo === build ===
cargo build --release --features cuda
if errorlevel 1 (
  echo Build failed. Nothing was staged, committed, or pushed.
  exit /b 1
)

echo === stage ===
git add -A
if errorlevel 1 exit /b 1
git status --short

git diff --cached --quiet
if errorlevel 1 (
  echo === commit ===
  echo %COMMIT_MSG%
  git commit -m "%COMMIT_MSG%"
  if errorlevel 1 (
    echo Commit failed. Nothing was pushed.
    exit /b 1
  )
) else (
  echo No staged changes. Skipping commit.
)

echo === push ===
git rev-parse --abbrev-ref "@{u}" >nul 2>&1
if errorlevel 1 (
  git push -u origin HEAD
) else (
  git push
)
if errorlevel 1 exit /b 1

echo Done.
exit /b 0
