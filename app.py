import streamlit as st
import core
import json
import os
import uuid
import time
from datetime import datetime

# 定数 define
HISTORY_FILE = "chat_history.json"

# ==========================================
# 関数定義: 履歴管理
# ==========================================
def load_all_history():
    """全セッションの履歴を読み込む"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_all_history(history_data):
    """全セッションの履歴を保存する"""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")

def create_new_session():
    """新しいセッションを作成する"""
    new_id = str(uuid.uuid4())
    st.session_state.current_session_id = new_id
    # 空のセッションデータを履歴に追加
    st.session_state.all_history[new_id] = {
        "title": "新しいチャット",
        "created_at": time.time(),
        "messages": []
    }
    return new_id

def delete_session(session_id):
    """セッションを削除する"""
    if session_id in st.session_state.all_history:
        del st.session_state.all_history[session_id]
        save_all_history(st.session_state.all_history)
        # カレントセッションだった場合は新規作成
        if st.session_state.current_session_id == session_id:
            create_new_session()
            st.rerun()

# ==========================================
# UI設定
# ==========================================
st.set_page_config(page_title="岩手県立大学AIチャット", page_icon="🏫", layout="wide")

# セッション状態の初期化
if "all_history" not in st.session_state:
    st.session_state.all_history = load_all_history()

# カレントセッションの初期化
if "current_session_id" not in st.session_state:
    # 履歴があれば最新のものを選択、なければ新規作成
    if st.session_state.all_history:
        # created_atでソートして最新を取得
        latest_id = sorted(
            st.session_state.all_history.items(),
            key=lambda x: x[1].get("created_at", 0),
            reverse=True
        )[0][0]
        st.session_state.current_session_id = latest_id
    else:
        create_new_session()

# エンジンの初期化 (変更なし)
if "engine" not in st.session_state:
    with st.spinner("システムを起動中... モデル読み込みとデータ収集を行っています (初回のみ時間がかかります)"):
        engine = core.RAGEngine()
        # 動作確認用にページ数を制限 (必要に応じて変更)
        engine.fetch_data(max_pages=133) 
        st.session_state.engine = engine
    st.success("準備完了！")

# ==========================================
# サイドバー: 履歴一覧
# ==========================================
with st.sidebar:
    st.title("🗂️ 履歴")
    if st.button("＋ 新しいチャット", use_container_width=True):
        create_new_session()
        st.rerun()
    
    st.divider()

    # 履歴リストを作成日時の降順でソート
    sorted_history = sorted(
        st.session_state.all_history.items(),
        key=lambda x: x[1].get("created_at", 0),
        reverse=True
    )

    for s_id, data in sorted_history:
        # ボタンのラベル (タイトルまたは日時)
        label = data.get("title", "新しいチャット")
        
        # 選択状態の強調
        if s_id == st.session_state.current_session_id:
            st.markdown(f"**👉 {label}**")
        else:
            if st.button(label, key=s_id, use_container_width=True):
                st.session_state.current_session_id = s_id
                st.rerun()

# ==========================================
# メインコンテンツ: チャットエリア
# ==========================================
current_id = st.session_state.current_session_id
current_session_data = st.session_state.all_history.get(current_id)

# 万が一IDが見つからない場合のフォールバック
if not current_session_data:
    create_new_session()
    current_id = st.session_state.current_session_id
    current_session_data = st.session_state.all_history[current_id]

messages = current_session_data["messages"]

# タイトル
st.title("🏫 岩手県立大学 RAGチャットボット")
st.caption(f"現在のセッション: {current_session_data['title']}")

# メッセージ表示
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "refs" in message and message["refs"]:
            st.markdown("**【参照リンク】**")
            for url in message["refs"]:
                st.markdown(f"- {url}")

# ユーザー入力
if prompt := st.chat_input("質問を入力してください..."):
    # ユーザーメッセージ表示
    st.chat_message("user").markdown(prompt)
    messages.append({"role": "user", "content": prompt})
    
    # 最初の質問の場合、タイトルを更新
    if len(messages) == 1:
        # タイトルを更新 (30文字制限)
        new_title = prompt[:20] + "..." if len(prompt) > 20 else prompt
        st.session_state.all_history[current_id]["title"] = new_title

    # 履歴保存 (即時反映)
    save_all_history(st.session_state.all_history)

    # AI回答生成
    with st.chat_message("assistant"):
        with st.spinner("回答を生成中..."):
            response_text, refs = st.session_state.engine.search(prompt)
            
            st.markdown(response_text)
            if refs:
                st.markdown("**【参照リンク】**")
                for url in refs:
                    st.markdown(f"- {url}")
            
    # メッセージ追加
    messages.append({"role": "assistant", "content": response_text, "refs": refs})
    save_all_history(st.session_state.all_history)
    st.rerun() # タイトル更新などを反映させるためリロード
