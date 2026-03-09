@echo off
REM 快速测试嵌入模型部署兼容性
REM 使用方法: 双击运行或在命令行中执行

echo 正在检查嵌入模型部署兼容性...
echo.

REM 检查Python是否可用
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: Python未安装或未添加到PATH
    echo 请先安装Python 3.8+
    pause
    exit /b 1
)

echo 正在运行兼容性检查...
echo.

REM 运行测试脚本
python test_embedding_model.py

echo.
echo 检查完成！
pause