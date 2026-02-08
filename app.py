import streamlit as st
import core

# ページ設定
st.set_page_config(page_title="岩手県立大学AIチャット", page_icon="🏫")

# タイトル
st.title("🏫 岩手県立大学 RAGチャットボット")
st.markdown("Python 3.10 + Llama.cpp + Streamlit 版")

# セッション状態でエンジンを保持
if "engine" not in st.session_state:
    with st.spinner("システムを起動中... モデル読み込みとデータ収集を行っています (初回のみ時間がかかります)"):
        engine = core.RAGEngine()
        engine.fetch_data(max_pages=20) # 動作確認用: 20ページ制限
        st.session_state.engine = engine
    st.success("準備完了！")

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 履歴の表示
for message in st.session_state.messages:
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
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI回答生成
    with st.chat_message("assistant"):
        with st.spinner("回答を生成中..."):
            response_text, refs = st.session_state.engine.search(prompt)
            
            st.markdown(response_text)
            if refs:
                st.markdown("**【参照リンク】**")
                for url in refs:
                    st.markdown(f"- {url}")
            
    # 履歴に追加
    st.session_state.messages.append({"role": "assistant", "content": response_text, "refs": refs})
