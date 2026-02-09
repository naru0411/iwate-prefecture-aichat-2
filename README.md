# 岩手県立大学 RAG チャットボット (v2)

Llama.cpp (Qwen2.5) を使用し、大学公式サイトの情報を元に回答するローカルRAGシステム。

## 特徴
- **簡単セットアップ**: 仮想環境（venv）とバッチファイルにより、複雑な手順なしで起動可能。
- **高精度な回答**: 133ページのスクレイピングデータに基づくコンテキスト活用。
- **高速起動**: ベクトルデータの永続化（storage保存）により、2回目以降は即座に開始。
- **履歴管理**: ChatGPT風のサイドバー履歴管理機能を搭載。

## セットアップ手順
1. **setup.bat** を実行（初回のみ）
   - 仮想環境の作成と、必要なライブラリのインストールを自動で行います。
2. **run.bat** を実行
   - アプリケーションが起動し、ブラウザでチャット画面が開きます。

## 使用技術
- **Language**: Python 3.10
- **UI**: Streamlit
- **Inference**: Llama-cpp-python (Model: Qwen2.5-1.5B-Instruct-GGUF)
- **Embeddings**: Sentence-Transformers (intfloat/multilingual-e5-small)

## ライセンス
[MIT License](LICENSE)
