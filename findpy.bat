@echo off
REM ===================================================================
REM  findpy.bat -- 被其他脚本 call 的小工具：找一个"真能跑"的 Python
REM  结果写入环境变量 PY
REM    PY_IS_RUNTIME=1  用的是项目自带运行时 runtime\（免安装，优先级最高）
REM    PY_IS_VENV=1     用的是项目 .venv
REM  注意：本文件故意不写 setlocal，否则变量传不回调用者。
REM ===================================================================
set "PY="
set "PY_IS_VENV="
set "PY_IS_RUNTIME="

REM 0) 项目自带运行时（解压 partC 即得；不需要装 Python、不需要联网）
if exist "runtime\python.exe" (
  "runtime\python.exe" -c "import numpy,torch,sentence_transformers,fastapi" >nul 2>nul
  if not errorlevel 1 (
    set "PY=%CD%\runtime\python.exe"
    set "PY_IS_RUNTIME=1"
  )
)

REM 1) 项目自带 .venv，但必须真的能启动（从别的电脑拷来的 .venv 会失效）
if not defined PY (
  if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "print(1)" >nul 2>nul
    if not errorlevel 1 (
      set "PY=%CD%\.venv\Scripts\python.exe"
      set "PY_IS_VENV=1"
    )
  )
)

REM 2) Windows 的 py 启动器
if not defined PY (
  py -3 -c "print(1)" >nul 2>nul
  if not errorlevel 1 for /f "delims=" %%I in ('py -3 -c "import sys;print(sys.executable)"') do set "PY=%%I"
)

REM 3) PATH 里的 python
if not defined PY (
  python -c "print(1)" >nul 2>nul
  if not errorlevel 1 for /f "delims=" %%I in ('python -c "import sys;print(sys.executable)"') do set "PY=%%I"
)

REM 4) WorkBuddy 自带的 Python（本机装过 WorkBuddy 一般会有）
if not defined PY (
  for /d %%D in ("%USERPROFILE%\.workbuddy\binaries\python\versions\*") do (
    if not defined PY (
      if exist "%%~fD\python.exe" (
        "%%~fD\python.exe" -c "print(1)" >nul 2>nul
        if not errorlevel 1 set "PY=%%~fD\python.exe"
      )
    )
  )
)

REM 5) 常见的本机安装位置（用户没勾 PATH 时兜底）
if not defined PY (
  for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "C:\Python313\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
  ) do (
    if not defined PY if exist %%P (
      %%P -c "print(1)" >nul 2>nul
      if not errorlevel 1 set "PY=%%~fP"
    )
  )
)

exit /b 0
