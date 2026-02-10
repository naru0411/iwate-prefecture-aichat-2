from core import RAGEngine
import sys
import time

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

def test_stream():
    engine = RAGEngine()
    # Use existing data (should be present if previous steps worked)
    # If not, it will default to empty searching or partial.
    # We just want to see if it yields.
    
    print("Searching...")
    # Mocking fetch if data missing? existing core.py handles loading.
    engine.fetch_data(max_pages=5) 
    
    stream, refs = engine.search("岩手県立大学について")
    
    print("Streaming start:")
    full_text = ""
    for chunk in stream:
        print(chunk, end="|", flush=True) # visual separator
        full_text += chunk
        time.sleep(0.05) # simulate delay if too fast
    
    print("\n\nFull Text:", full_text)
    print("Refs:", refs)

if __name__ == "__main__":
    test_stream()
