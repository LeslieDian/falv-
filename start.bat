@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ===============================================================
echo   法律领域智能体 · 本地演示
echo ===============================================================
echo.

call "%~dp0findpy.bat"
if not defined PY (
  echo [错误] 这台电脑上还没有可用的运行环境。
  echo        请先双击 deploy.bat（会自动选"自带运行时/离线装/联网装"三条路）。
  echo.
  pause
  exit /b 1
)
if defined PY_IS_RUNTIME echo 运行时：自带 runtime\（免安装）
if defined PY_IS_VENV echo 运行时：项目 .venv

if not exist "kb\vectors.f16.npy" (
  echo.
  echo [错误] 缺 kb\ 知识库 —— 请先解压 partA 包到本目录。
  echo.
  pause
  exit /b 1
)

echo.
echo 服务正在启动（加载知识库与编码器约需 30-60 秒）。
echo 浏览器会在 15 秒后自动打开 http://127.0.0.1:8760
echo 本窗口不要关闭；按 Ctrl+C 或直接关窗口即停止服务。
echo.
start "" /min cmd /c "timeout /t 15 /nobreak > nul & start http://127.0.0.1:8760"
echo ---------------------------------------------------------------
"%PY%" run.py
echo ---------------------------------------------------------------
echo 服务已停止。
pause
