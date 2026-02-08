import os
import re
import time
import requests
import numpy as np
import urllib3
import pickle
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from markdownify import markdownify as md
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# SSL証明書エラーの警告を非表示
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 1. モデル設定 & 定数
# ==========================================
MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# ストレージ設定
STORAGE_DIR = "storage"
DOCS_FILE = os.path.join(STORAGE_DIR, "documents.pkl")
META_FILE = os.path.join(STORAGE_DIR, "metadatas.pkl")
EMBED_FILE = os.path.join(STORAGE_DIR, "embeddings.npy")

# ==========================================
# 2. クラス定義: RAG エンジン
# ==========================================
class RAGEngine:
    def __init__(self):
        print("システムを初期化中...")
        self.llm = self._load_llm()
        self.embed_model = self._load_embed_model()
        self.documents = []
        self.document_embeddings = None
        self.metadatas = []
        
    def _load_llm(self):
        """Llamaモデルをロード"""
        # ローカルディレクトリ ./models に保存
        print(f"モデル {MODEL_REPO} をダウンロード/ロード中...")
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir="./models",
            local_dir_use_symlinks=False
        )
        print(f"モデルロード: {model_path}")
        # n_gpu_layers=-1 で可能な限りGPU使用, n_ctx=2048
        return Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=-1,
            verbose=False
        )

    def _load_embed_model(self):
        """埋め込みモデルをロード (CPU設定)"""
        print(f"Embeddingモデル {EMBEDDING_MODEL_NAME} をロード中 (CPU)...")
        return SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

    def _save_data(self):
        """データをストレージに保存"""
        if not os.path.exists(STORAGE_DIR):
            os.makedirs(STORAGE_DIR)
        
        try:
            with open(DOCS_FILE, 'wb') as f:
                pickle.dump(self.documents, f)
            with open(META_FILE, 'wb') as f:
                pickle.dump(self.metadatas, f)
            np.save(EMBED_FILE, self.document_embeddings)
            print("データをlocal storageに保存しました。")
        except Exception as e:
            print(f"保存エラー: {e}")

    def _load_data(self):
        """ストレージからデータを読み込み"""
        if os.path.exists(DOCS_FILE) and os.path.exists(META_FILE) and os.path.exists(EMBED_FILE):
            print("保存済みデータが見つかりました。読み込んでいます...")
            try:
                with open(DOCS_FILE, 'rb') as f:
                    self.documents = pickle.load(f)
                with open(META_FILE, 'rb') as f:
                    self.metadatas = pickle.load(f)
                self.document_embeddings = np.load(EMBED_FILE)
                print("データの読み込み完了。スクレイピングをスキップします。")
                return True
            except Exception as e:
                print(f"読み込みエラー: {e}")
                return False
        return False

    def fetch_data(self, base_url: str="https://www.iwate-pu.ac.jp/", max_pages: int=133):
        """大学サイトからデータを収集 (保存済みならスキップ)"""

        # 既存データの読み込み試行
        if self._load_data():
            return
            
        print(f"データ収集中 (最大 {max_pages} ページ)...")
        results = []
        visited_urls = set()
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

        with requests.Session() as s:
            s.headers.update(headers)
            target_links = []

            # 重要ページ
            important_urls = [
                "https://www.iwate-pu.ac.jp/faculty/",
                "https://www.iwate-pu.ac.jp/examination/all.html"
            ]
            for imp_url in important_urls:
                target_links.append({"url": imp_url, "title": "重要ページ"})
                visited_urls.add(imp_url)

            # トップページからリンク収集
            try:
                response = s.get(base_url, timeout=10, verify=False)
                response.encoding = response.apparent_encoding
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.find_all("a", href=True):
                    full_url = urljoin(base_url, link["href"])
                    if ("iwate-pu.ac.jp" in full_url) and \
                       (full_url not in visited_urls) and \
                       (not full_url.endswith((".jpg", ".png", ".pdf", ".zip", ".css", ".js"))) and \
                       ("#" not in full_url):
                        target_links.append({"url": full_url, "title": link.text.strip()})
                        visited_urls.add(full_url)
            except Exception as e:
                print(f"Error crawling top: {e}")

            # 各ページの中身を取得
            count = 0
            for target in target_links:
                if count >= max_pages: break
                try:
                    res = s.get(target["url"], timeout=10, verify=False)
                    res.encoding = res.apparent_encoding
                    soup = BeautifulSoup(res.text, "html.parser")
                    title = soup.title.text.strip() if soup.title else target["title"]

                    # ノイズ除去
                    noise_selectors = [
                        'header', 'footer', 'nav', 'noscript', 'script', 'style',
                        '.topic_path', '.pankuzu', '#pan', '.side_menu', '.search_area'
                    ]
                    for selector in noise_selectors:
                        for noise in soup.select(selector):
                            noise.decompose()

                    main_content = soup.find(id="contents") or soup.find("main") or soup.body
                    if main_content:
                        raw_text = md(str(main_content), strip=['a', 'img', 'script', 'style', 'iframe'])

                        # 行単位クリーニング
                        lines = raw_text.split('\n')
                        cleaned_lines = []
                        for line in lines:
                            line = line.strip()
                            if not line: continue
                            if line in ["ホーム", "Home", "TOP", "学部・大学院等"]: continue
                            if line.startswith(">") or re.match(r'^[-=]{3,}$', line): continue
                            cleaned_lines.append(line)

                        content_text = "\n".join(cleaned_lines)
                        content_text = re.sub(r'\n{3,}', '\n\n', content_text)

                        if len(content_text) > 50:
                            results.append({"title": title, "url": target["url"], "content": content_text})
                            count += 1
                            time.sleep(0.1)
                except Exception:
                    pass
        
        print(f"データ収集完了: {len(results)} ページ")
        self._process_documents(results)
        self._save_data() # Save after processing

    def _process_documents(self, raw_data, chunk_size=500):
        """データをチャンク化してベクトル化"""
        print("データをチャンク化中...")
        self.documents = []
        self.metadatas = []

        for item in raw_data:
            title = item["title"]
            url = item["url"]
            content = item["content"]
            
            # 簡易的なチャンク分割 (文字数ベースで近似)
            source_str = f" (出典: {title})"
            current_chunk = ""
            
            # 文単位で分割
            sentences = content.replace("。", "。\n").split("\n")
            
            for sentence in sentences:
                if not sentence.strip(): continue
                # およそのトークン数見積もり (文字数)
                if len(current_chunk) + len(sentence) > chunk_size:
                    if current_chunk:
                        self.documents.append(current_chunk + source_str)
                        self.metadatas.append({"title": title, "url": url})
                    current_chunk = sentence
                else:
                    current_chunk += sentence
            
            if current_chunk:
                self.documents.append(current_chunk + source_str)
                self.metadatas.append({"title": title, "url": url})

        print(f"ベクトル化を実行中 ({len(self.documents)} チャンク)...")
        # prefix "passage: " は e5-small の推奨
        docs_for_embed = ["passage: " + d for d in self.documents]
        self.document_embeddings = self.embed_model.encode(docs_for_embed, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    def search(self, query: str, top_k: int=3):
        """検索と生成"""
        if not self.documents:
            return "データがロードされていません。", []

        # クエリベクトル化 (prefix "query: " は e5-small の推奨)
        query_vec = self.embed_model.encode(["query: " + query], normalize_embeddings=True)[0]
        
        # 類似度計算
        scores = cosine_similarity([query_vec], self.document_embeddings)[0]

        # スコアブースト (えこひいき)
        BOOST_URL_KEYWORD = ("faculty", "examination/all.html", "access")
        BOOST_FACTOR = 1.2
        
        for i, meta in enumerate(self.metadatas):
            if any(keyword in meta['url'] for keyword in BOOST_URL_KEYWORD):
                scores[i] *= BOOST_FACTOR

        top_indices = scores.argsort()[::-1][:top_k]
        
        context_texts = []
        ref_urls = []
        seen_urls = set()

        for idx in top_indices:
            text = self.documents[idx]
            meta = self.metadatas[idx]
            context_texts.append(f"【資料】(出典:{meta['title']})\n{text}")
            if meta['url'] not in seen_urls:
                ref_urls.append(meta['url'])
                seen_urls.add(meta['url'])

        combined_context = "\n\n".join(context_texts)
        
        # プロンプト作成
        system_content = """あなたは岩手県立大学の広報アシスタントAIです。
以下の【参照資料】の内容を統合して、ユーザーの【質問】に詳しく答えてください。
回答にはURLを含めず、嘘を書かないでください。
情報がない場合は『NO_INFO』とだけ出力してください。"""
        
        user_content = f"""### 質問
{query}

### 参照資料
{combined_context}

### 回答
"""
        
        # Llama推論
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            max_tokens=1024
        )
        
        answer = response["choices"][0]["message"]["content"]
        
        if "NO_INFO" in answer:
            answer = "申し訳ありません。現時点の資料には詳しい記載がありませんでした。\n回答に近いと思われる以下のWebページをご確認いただけますでしょうか。"
            
        return answer, ref_urls

# シングルトンインスタンス作成用関数
def get_engine():
    if 'engine' not in  globals():
        globals()['engine'] = RAGEngine()
    return globals()['engine']

if __name__ == "__main__":
    # テスト実行用
    engine = RAGEngine()
    engine.fetch_data()
    ans, refs = engine.search("学部について教えて")
    print(ans)
    print(refs)
