
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
                data = json.load(f)
                # 辞書型でない場合（古いリスト形式など）は空の辞書を返す
                if not isinstance(data, dict):
                    return {}
                return data
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
        
        # カレントセッションだった場合
        if st.session_state.current_session_id == session_id:
            # 他に履歴があれば最新に切り替え、なければ新規作成
            if st.session_state.all_history:
                latest_id = sorted(
                    st.session_state.all_history.items(),
                    key=lambda x: x[1].get("created_at", 0),
                    reverse=True
                )[0][0]
                st.session_state.current_session_id = latest_id
            else:
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

    # スタイル調整: サイドバーのボタン間の余白を詰め、配置を整える
    st.markdown("""
        <style>
        div[data-testid="stSidebarUserContent"] .stButton button {
            margin-bottom: -10px;
        }
        div[data-testid="stSidebarUserContent"] .stPopover button {
            padding-top: 0;
            padding-bottom: 0;
            height: 38px;
        }
        </style>
    """, unsafe_allow_html=True)

    for s_id, data in sorted_history:
        label = data.get("title", "新しいチャット")
        
        # UIレイアウト: ボタンと設定メニューを中央揃えで配置
        col1, col2 = st.columns([0.8, 0.2], vertical_alignment="center")
        
        # チャット選択ボタン
        with col1:
            display_label = f"👉 {label}" if s_id == st.session_state.current_session_id else label
            if st.button(display_label, key=f"sel_{s_id}", use_container_width=True):
                if s_id != st.session_state.current_session_id:
                    st.session_state.current_session_id = s_id
                    st.rerun()
        
        # 設定メニュー (ポップオーバー)
        with col2:
            with st.popover("⋮", use_container_width=True):
                st.markdown("##### 設定")
                
                # 【名前の変更】
                new_title = st.text_input("タイトル変更", value=label, key=f"rename_{s_id}")
                if st.button("保存", key=f"save_rename_{s_id}"):
                    if new_title.strip():
                        st.session_state.all_history[s_id]["title"] = new_title
                        save_all_history(st.session_state.all_history)
                        st.rerun()
                
                st.divider()
                
                # 【削除】
                if st.button("🗑️ 削除", key=f"del_{s_id}", type="primary", use_container_width=True):
                    delete_session(s_id)

# ==========================================
# メインコンテンツ: チャットエリア
# ==========================================
current_id = st.session_state.current_session_id
current_session_data = st.session_state.all_history.get(current_id)

# 万が一IDが見つからない場合のフォールバック
if not current_session_data:
    # 履歴が空なら新規作成
    if not st.session_state.all_history:
        create_new_session()
    else:
        # IDが無効な場合、最新に戻す
        latest_id = sorted(
            st.session_state.all_history.items(),
            key=lambda x: x[1].get("created_at", 0),
            reverse=True
        )[0][0]
        st.session_state.current_session_id = latest_id
    
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
        save_all_history(st.session_state.all_history) # 即時保存

    # 履歴保存 (即時反映)
    save_all_history(st.session_state.all_history)

    # AI回答生成
    with st.chat_message("assistant"):
        # 検索処理中はスピナーを表示
        with st.spinner("回答を検索・生成中..."):
            stream_gen, refs = st.session_state.engine.search(prompt)
            
        # ストリーミング表示 (st.write_streamはGeneratorを受け取り、完了後の全文を返す)
        response_text = st.write_stream(stream_gen)
        
        if refs:
            st.markdown("**【参照リンク】**")
            for url in refs:
                st.markdown(f"- {url}")
            
    # メッセージ追加
    messages.append({"role": "assistant", "content": response_text, "refs": refs})
    save_all_history(st.session_state.all_history)
    st.rerun() # タイトル更新などを反映させるためリロード
