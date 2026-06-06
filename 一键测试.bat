@echo off
chcp 65001 >nul
echo ============================================
echo   智能病历系统 - 一键全量测试
echo ============================================
echo.

cd /d "%~dp0backend"

echo [1/2] 设置编码...
set PYTHONIOENCODING=utf-8

echo [2/2] 运行全部测试...
echo.
python tests\run_all_tests.py

if %errorlevel% equ 0 (
    echo.
    echo ============================================
    echo   🎉 全部测试通过！
    echo ============================================
) else (
    echo.
    echo ============================================
    echo   ❌ 部分测试失败
    echo ============================================
)

echo.
pause