@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal
echo ===============================================================
echo   一键部署：让这台电脑能跑起本地演示
echo ===============================================================
echo   三种模式（脚本自动选，越靠前越省事）：
echo     1) 有 runtime\  免安装免联网，直接开工
echo     2) 有 wheels\  离线装依赖，仍然不需要联网
echo     3) 都没有      从国内公开镜像装（阿里云 / pytorch / hf-mirror）
echo   注意：这里说的"联网"永远是公网，跟公司内网服务器无关。
echo ===============================================================
echo.

call "%~dp0findpy.bat"

REM ---------- 模式 1：自带运行时，已经万事俱备 ----------
if defined PY_IS_RUNTIME (
  echo [OK] 检测到自带运行时 runtime\python.exe
  echo      本机无需安装 Python、无需 pip、无需联网，直接进入自检。
  echo.
  goto :SELFTEST
)

if not defined PY (
  echo [错误] 没有找到任何 Python 解释器。
  echo.
  echo   两条出路，任选其一：
  echo     A. 解压 partC（自带运行时）到本目录，再双击本文件 —— 不用装 Python
  echo     B. 自己装一个 Python 3.10+：https://www.python.org/downloads/
  echo        安装时务必勾选 "Add python.exe to PATH"，装好后重新双击本文件
  echo.
  pause
  exit /b 1
)

if defined PY_IS_VENV (
  echo [OK] 已检测到可用的 .venv，跳过安装步骤。
  echo.
  goto :MODEL
)

echo 基础解释器：%PY%
"%PY%" -c "import sys;sys.exit(0 if sys.version_info>=(3,9) else 1)"
if errorlevel 1 (
  echo [错误] Python 版本过低，请安装 3.10 或更高版本。
  pause
  exit /b 1
)
if exist ".venv" (
  echo [提示] 检测到不可用的 .venv（虚拟环境不能跨电脑拷贝），正在删除重建...
  rmdir /s /q ".venv"
)
echo [1/2] 创建虚拟环境 .venv ...
"%PY%" -m venv ".venv"
if not exist ".venv\Scripts\python.exe" (
  echo [错误] 虚拟环境创建失败。
  pause
  exit /b 1
)
set "PY=%CD%\.venv\Scripts\python.exe"
echo.

REM ---------- 模式 2：本地有离线包，零联网安装 ----------
if exist "wheels" (
  echo [2/2] 检测到 wheels\ 离线依赖包 —— 本次安装全程不联网。
  echo       安装 Web 与检索依赖 ...
  "%PY%" -m pip install --no-index --no-warn-script-location ^
      --find-links="%CD%\wheels" fastapi uvicorn httpx pydantic numpy torch sentence-transformers
  if errorlevel 1 (
    echo [错误] 离线安装失败。请确认 wheels\ 已完整解压（应含 50 个左右 .whl）。
    pause
    exit /b 1
  )
  echo.
  goto :MODEL
)

REM ---------- 模式 3：公网在线安装 ----------
echo [2/2] 从国内公开镜像安装依赖（约 400MB，最慢的一步）...
echo       若这一步失败，别急：改用离线包重跑即可 ——
echo       把 partC 解压出的 wheels\ 放进本目录，再双击 deploy.bat。
echo.
echo       安装 pip 与 Web 依赖 ...
"%PY%" -m pip install -U pip -i https://mirrors.aliyun.com/pypi/simple/
"%PY%" -m pip install -i https://mirrors.aliyun.com/pypi/simple/ fastapi uvicorn httpx pydantic numpy
echo       安装 sentence-transformers ...
"%PY%" -m pip install -i https://mirrors.aliyun.com/pypi/simple/ sentence-transformers
echo       安装 torch（CPU 版，约 200MB，最慢的一步）...
"%PY%" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
if errorlevel 1 (
  echo       [提示] pytorch 官方源不通，改走阿里云 PyPI 重试 torch ...
  "%PY%" -m pip install -i https://mirrors.aliyun.com/pypi/simple/ torch
)
echo.

:MODEL
echo ---------------------------------------------------------------
echo   检查知识库与查询编码器
echo ---------------------------------------------------------------
if not exist "kb\vectors.f16.npy" (
  echo [错误] 缺 kb\ 知识库 —— 请先解压 partA 包到本目录。
  pause
  exit /b 1
)
echo [OK] 知识库已就位。
if exist "models\Qwen3-Embedding-0.6B\model.safetensors" (
  echo [OK] 查询编码器已就位（models\Qwen3-Embedding-0.6B）。
) else (
  echo [缺少] 查询编码器（约 1.15GB）。检索必须用它把问题编码成向量。
  echo        任选一种补上：
  echo          1. 解压 partB 包，得到 models\Qwen3-Embedding-0.6B\
  echo          2. 联网时跑：  "%PY%" -X utf8 fetch_model.py
  echo.
  echo        【重要】缺它就只能看知识库，不能提问。请先补上再继续。
  pause
  exit /b 1
)
echo.

:SELFTEST
echo ---------------------------------------------------------------
echo   自检（不联网：验证知识库 + 编码器 + 检索链路）
echo ---------------------------------------------------------------
"%PY%" -X utf8 "%~dp0selftest.py"
echo.

echo ---------------------------------------------------------------
echo   完整体检
echo ---------------------------------------------------------------
"%PY%" -X utf8 "%~dp0doctor.py"
echo.

echo ===============================================================
echo   部署结束。下一步：双击 start.bat，浏览器打开 http://127.0.0.1:8760
echo.
echo   关于联网，只有一件事需要知道：
echo     检索层 —— 纯本地，断网照常出法条与案例
echo     生成层 —— 要一份大模型 API 的 key（config.json 里填）
echo              没网 / 没 key 时会自动降级为"本地检索模式"，照样能演示
echo   公司内网服务器与答辩演示无关，不在本链路的任何一环。
echo ===============================================================
pause
