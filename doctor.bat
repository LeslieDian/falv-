@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ===============================================================
echo   体检：为什么打不开 / 跑不起来？
echo ===============================================================
call "%~dp0findpy.bat"
if not defined PY (
  echo.
  echo [错误] 这台电脑上找不到可用的 Python 解释器。
  echo        请先安装 Python 3.10 或更高版本：
  echo            https://www.python.org/downloads/
  echo        安装时务必勾选 "Add python.exe to PATH"，
  echo        装好后重新双击本文件。
  echo.
  pause
  exit /b 1
)
echo 使用的解释器：%PY%
echo.
"%PY%" -X utf8 "%~dp0doctor.py" %*
echo.
pause
