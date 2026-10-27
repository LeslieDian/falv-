@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal
echo ===============================================================
echo   打包给答辩电脑（只拷必需文件，约 1.9 GB）
echo ===============================================================
echo.
echo 把代码 + kb/ + models/ + router/ + web/ 拷到目标目录；
echo 自动排除 .venv（虚拟环境不能跨电脑用，新机由 deploy.bat 重建）。
echo.
echo 目标目录示例：
echo     E:\law-rag              （U 盘根目录，会自动创建文件夹）
echo     D:\dabian\law-rag
echo.
set /p DEST=请输入目标目录：
if "%DEST%"=="" (
  echo 未输入，已取消。
  pause
  exit /b 1
)
echo.
robocopy "%~dp0." "%DEST%" /E /XD ".venv" "__pycache__" ".git" /XF "_*" /NFL /NDL /NJH /NP /R:1 /W:1
set RC=%ERRORLEVEL%
if %RC% GEQ 8 (
  echo.
  echo [错误] 复制失败（robocopy 代码 %RC%）。
  echo        常见原因：目标盘写保护 / 空间不足 / 路径不存在。
  pause
  exit /b 1
)
echo.
echo 复制完成，正在对目标目录做体检（含 MD5 校验，约 10 秒）...
call "%~dp0findpy.bat"
if defined PY "%PY%" -X utf8 "%~dp0doctor.py" --root "%DEST%"
echo.
echo ===============================================================
echo   到了新电脑上，三步：
echo     1) 双击 deploy.bat   （装环境，需要联网）
echo     2) 双击 start.bat    （启动服务）
echo     3) 浏览器打开 http://127.0.0.1:8760
echo   有问题就双击 doctor.bat 看体检报告。
echo ===============================================================
pause
