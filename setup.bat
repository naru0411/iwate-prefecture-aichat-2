@echo off
chcp 65001
echo ==========================================
echo 岩手県立大学AIチャット セットアップスクリプト
echo ==========================================

cd /d %~dp0

if not exist ".venv" (
    echo [INFO] 仮想環境(.venv)を作成しています...
    python -m venv .venv
) else (
    echo [INFO] 既存の仮想環境を使用します。
)

echo [INFO] 仮想環境を有効化しています...
call .venv\Scripts\activate

echo [INFO] pipを更新しています...
python -m pip install --upgrade pip

echo [INFO] ライブラリをインストールしています...
pip install -r requirements.txt

echo.
echo ==========================================
echo セットアップが完了しました！
echo run.bat を実行してチャットを開始してください。
echo ==========================================
pause
