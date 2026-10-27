@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 说明：安装步骤已并入 deploy.bat（会把依赖安装 + 自检 + 体检一次做完）。
echo.
call "%~dp0deploy.bat" %*
