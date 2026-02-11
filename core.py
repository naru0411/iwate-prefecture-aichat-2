import os
import re
import time
import requests
import numpy as np
import urllib3
import pickle
from bs4 import BeautifulSoup, Tag
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
PARENTS_FILE = os.path.join(STORAGE_DIR, "parents.pkl")

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
        self.parent_documents = []
        
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
            n_ctx=4096,
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
            with open(PARENTS_FILE, 'wb') as f:
                pickle.dump(self.parent_documents, f)
            np.save(EMBED_FILE, self.document_embeddings)
            print("データをlocal storageに保存しました。")
        except Exception as e:
            print(f"保存エラー: {e}")

    def _load_data(self):
        """ストレージからデータを読み込み"""
        if os.path.exists(DOCS_FILE) and os.path.exists(META_FILE) and os.path.exists(EMBED_FILE) and os.path.exists(PARENTS_FILE):
            print("保存済みデータが見つかりました。読み込んでいます...")
            try:
                with open(DOCS_FILE, 'rb') as f:
                    self.documents = pickle.load(f)
                with open(META_FILE, 'rb') as f:
                    self.metadatas = pickle.load(f)
                with open(PARENTS_FILE, 'rb') as f:
                    self.parent_documents = pickle.load(f)
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
                        # Table処理: 構造を維持してテキスト化
                        for table in main_content.find_all("table"):
                            rows_text = []
                            for tr in table.find_all("tr"):
                                cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
                                if not cells: continue
                                # 列数に応じてフォーマット変更
                                if len(cells) == 2:
                                    rows_text.append(f"{cells[0]}: {cells[1]}")
                                else:
                                    rows_text.append(" | ".join(cells))
                            if rows_text:
                                table.replace_with("\n" + "\n".join(rows_text) + "\n")

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
                        
                        # 親子整合性のため、ここで「。」を改行コード付きに変換しておく
                        content_text = content_text.replace("。", "。\n")

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
        self.parent_documents = []

        for item in raw_data:
            title = item["title"]
            url = item["url"]
            content = item["content"] # 既に "。\n" が適用済み
            
            # 親ドキュメントとして保存
            self.parent_documents.append(content)
            parent_id = len(self.parent_documents) - 1
            
            source_str = f" (出典: {title})"
            current_chunk = ""
            current_start_idx = 0
            # 現在のチャンクの開始位置を追跡するためのオフセット
            cursor = 0 
            
            # 文単位で分割 (改行コードで分割すればよい)
            sentences = content.split("\n")
            
            for sentence in sentences:
                # splitで消えた改行を長さ計算に含める必要があるが、
                # contentには \n が入っている。splitすると \n は消える。
                # よって len(sentence) + 1 (for \n) が元の長さ。
                # ただし最後の行など注意。
                
                # 正確な位置特定のため、findを使う方が安全か、
                # あるいは単純に積み上げるか。
                # ここでは積み上げ方式で行く (Approximation)
                sent_len = len(sentence) + 1 # +1 for newline
                
                if not sentence.strip(): 
                    cursor += sent_len
                    continue

                if len(current_chunk) + len(sentence) > chunk_size:
                    if current_chunk:
                        self.documents.append(current_chunk + source_str)
                        self.metadatas.append({
                            "title": title, 
                            "url": url,
                            "parent_id": parent_id,
                            "start_index": current_start_idx,
                            "end_index": cursor
                        })
                    current_chunk = sentence
                    current_start_idx = cursor # 次のチャンクの開始位置
                else:
                    if current_chunk:
                        current_chunk += "\n" + sentence
                    else:
                        current_chunk = sentence
                
                cursor += sent_len
            
            if current_chunk:
                self.documents.append(current_chunk + source_str)
                self.metadatas.append({
                    "title": title, 
                    "url": url,
                    "parent_id": parent_id,
                    "start_index": current_start_idx,
                    "end_index": cursor
                })

        print(f"ベクトル化を実行中 ({len(self.documents)} チャンク)...")
        # prefix "passage: " は e5-small の推奨
        docs_for_embed = ["passage: " + d for d in self.documents]
        self.document_embeddings = self.embed_model.encode(docs_for_embed, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    def search(self, query: str, top_k: int=3):
        """検索と生成"""
        if not self.documents:
            return "データがロードされていません。", []

        # クエリベクトル化
        query_vec = self.embed_model.encode(["query: " + query], normalize_embeddings=True)[0]
        
        # 類似度計算
        scores = cosine_similarity([query_vec], self.document_embeddings)[0]

        # スコアブースト
        BOOST_URL_KEYWORD = ("faculty", "examination/all.html", "access")
        BOOST_FACTOR = 1.1
        
        for i, meta in enumerate(self.metadatas):
            if any(keyword in meta['url'] for keyword in BOOST_URL_KEYWORD):
                scores[i] *= BOOST_FACTOR

        top_indices = scores.argsort()[::-1][:top_k]
        
        context_texts = []
        ref_urls = []
        seen_urls = set()

        for idx in top_indices:
            meta = self.metadatas[idx]
            
            # Parent-Document Retrieval (Window)
            parent_id = meta.get("parent_id")
            if parent_id is not None and 0 <= parent_id < len(self.parent_documents):
                parent_text = self.parent_documents[parent_id]
                start_window = max(0, meta["start_index"] - 500)
                end_window = min(len(parent_text), meta["end_index"] + 500)
                
                # Window抽出
                window_text = parent_text[start_window:end_window]
                context_texts.append(f"【資料】(出典:{meta['title']})\n...{window_text}...")
            else:
                # Fallback: keep existing behavior if parent not found (should not happen)
                context_texts.append(f"【資料】(出典:{meta['title']})\n{self.documents[idx]}")

            if meta['url'] not in seen_urls:
                ref_urls.append(meta['url'])
                seen_urls.add(meta['url'])

        combined_context = "\n\n".join(context_texts)
        
        # 生成ロジック呼び出し
        return self._generate_answer(query, combined_context, ref_urls)


    def _generate_answer(self, query, context, ref_urls):
        """厳密な生成とハルシネーション抑制ロジック (超厳格モード・3B対応版)"""
        
        system_prompt = """あなたは岩手県立大学の厳格な検証官です。以下の【禁止事項】を犯した場合、回答は失敗とみなされます。

【禁止事項】
1. **日付の論理矛盾**: 「10月下旬（1月26日）」のように、同一文脈内で月が一致しない、または時系列が破綻している記述は即座に検知し、その回答を放棄して『NO_INFO』と出力せよ。3Bモデルの推論能力を最大限に活かし、矛盾を見逃すな。
2. **固有名詞の改変禁止**: 資料にある学部名、学科名、コース名、資格名などは、一言一句たりとも改変してはならない。「ソフトウェア」を「软件」とするような誤変換は論外である。
3. **表データの捏造**: 行と列の対応関係が100%確実でない限り、表から情報を読み取るな。

【鉄則】
1. **情報の分離**: 資料内に『質問された学部以外の名前』が出てきた場合、その前後にある情報はすべて無視せよ。
2. **自己検閲**: 回答を生成した後、一度立ち止まって「この日付に矛盾はないか？」「固有名詞は正しいか？」を自問自答せよ。少しでも疑わしければ『NO_INFO』とせよ。"""
        
        user_prompt = f"""### 参照資料
{context}

### 質問
{query}

### 回答"""

        # Llama推論 (Fact-based設定)
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,       # ランダム性を排除
            top_p=0.85,           # 低確率の単語をカット
            repeat_penalty=1.1,   # 繰り返しを抑制
            max_tokens=512
        )
        
        answer = response["choices"][0]["message"]["content"].strip()
        
        # --- ポストプロセス (強化版) ---
        
        # 1. 中国語の強制置換 (念のため維持)
        answer = answer.replace("软件", "ソフトウェア")
        
        # 2. 矛盾検知 (日付の混同)
        import re
        # パターンA: 括弧内の矛盾 (例: 10月(1月))
        date_contradiction = re.search(r'(\d+)月[^。、]*[（(].*?(\d+)月', answer)
        if date_contradiction:
            m1, m2 = date_contradiction.groups()
            if m1 != m2:
                return self._get_fallback_message(), ref_urls

        # パターンB: 離れた月の矛盾 (例: 9月...1月...)
        months = [int(m) for m in re.findall(r'(\d+)月', answer)]
        if len(set(months)) >= 2:
            if 9 in months and 1 in months:
                 return self._get_fallback_message(), ref_urls

        # 3. NO_INFO チェック
        if "NO_INFO" in answer or len(answer) < 5:
            return self._get_fallback_message(), ref_urls

        return answer, ref_urls

    def _get_fallback_message(self):
        return "申し訳ありません。提供された資料からは、情報の混同を避けつつ正確な回答を作成することができませんでした。間違いを防ぐため、以下の公式サイトより最新の情報を直接ご確認ください。"

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
