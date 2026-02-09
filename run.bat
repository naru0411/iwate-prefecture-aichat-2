@echo off
chcp 65001
cd /d %~dp0

if not exist ".venv" (
    echo [ERROR] 仮想環境が見つかりません。先に setup.bat を実行してください。
    pause
    exit /b
)

echo [INFO] 仮想環境を有効化して起動します...
call .venv\Scripts\activate

echo [INFO] アプリケーションを起動しています...
streamlit run app.py

pause
