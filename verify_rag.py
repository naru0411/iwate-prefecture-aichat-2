from core import RAGEngine
import sys
import os

# Avoid encoding errors
sys.stdout.reconfigure(encoding='utf-8')

def test_rag():
    # Clean storage for fresh test (optional, but good for verification)
    # create instance
    engine = RAGEngine()
    
    # Fetch data (limited pages)
    print("Fetching data (limit 10)...")
    engine.fetch_data(base_url="https://www.iwate-pu.ac.jp/", max_pages=10)
    
    # Check data integrity
    print(f"Documents: {len(engine.documents)}")
    print(f"Parent Docs: {len(engine.parent_documents)}")
    
    if len(engine.parent_documents) > 0:
        print(f"Sample parent doc length: {len(engine.parent_documents[0])}")
    
    # Check Search
    rec_queries = [
        "入試日程について教えて",
        "ソフトウェア情報学部のアドミッションポリシーは？",
        "学長は誰ですか？"
    ]
    
    for q in rec_queries:
        print(f"\nQuery: {q}")
        ans, refs = engine.search(q, top_k=3)
        print(f"Answer: {ans}")
        print(f"Refs: {refs}")

if __name__ == "__main__":
    test_rag()
