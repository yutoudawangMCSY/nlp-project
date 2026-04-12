@echo off
chcp 65001 >nul
set "ROOT=%~dp0"
cd /d "%ROOT%"

python preprocess_and_train_step1_from_files.py ^
  --data_root "E:\迅雷下载\CMDA_管理层讨论与分析_ALL" ^
  --text_subdir "文本" ^
  --year_start 2022 ^
  --year_end 2022 ^
  --k_min 20 ^
  --k_max 20 ^
  --k_step 1

if errorlevel 1 (
  echo.
  echo [错误] 运行失败，请确认已安装依赖: pip install -r requirements.txt
  pause
  exit /b 1
)

echo.
echo [完成] 输出目录（默认）: outputs\step1_from_files\yearly_lda
pause
