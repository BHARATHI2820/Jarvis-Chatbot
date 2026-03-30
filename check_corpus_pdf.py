# ============================================================
# SCRIPT: check_corpus.py
# Purpose: Inspect what's stored in your corpus memory
# what stored inside the file 

# index.pkl contains:

# ✅ Document content (questions and answers text)
# ✅ Metadata (timestamps, question text)
# ✅ Document IDs (UUIDs like 63f66bec-a7dd-411c-8e2c-304b16c28ee3)
# ✅ ID mappings (linking vector index to document IDs)

# index.faiss contains:

# ❌ Only numerical vectors (embeddings)
# ❌ No text, no answers, no questions
# Just 384-dimensional numbers like [-0.045, -0.028, 0.044, ...]
# ============================================================

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import json
from datetime import datetime

# Setup
CORPUS_FOLDER = "faiss_store/pdf_documents"
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Load corpus
try:
    vectorstore = FAISS.load_local(
        CORPUS_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✅ Loaded corpus successfully!")
    print(f"📊 Total entries: {vectorstore.index.ntotal}")
    print("=" * 60)
    
    # Access the document store
    docs_dict = vectorstore.docstore._dict
    
    print(f"\n📚 Corpus Contents:\n")
    
    for i, (doc_id, doc) in enumerate(docs_dict.items(), 1):
        print(f"\n{'='*60}")
        print(f"Entry #{i}")
        print(f"{'='*60}")
        
        # Get metadata
        question = doc.metadata.get('question', 'N/A')
        timestamp = doc.metadata.get('timestamp', 'N/A')
        
        # Get full text
        full_text = doc.page_content
        
        # Split into question and answer
        if "ANSWER:\n" in full_text:
            parts = full_text.split("ANSWER:\n", 1)
            saved_question = parts[0].replace("QUESTION: ", "").strip()
            answer = parts[1].strip()
        else:
            saved_question = question
            answer = full_text
        
        # Display
        print(f"📝 Question: {saved_question}")
        print(f"🕐 Saved at: {timestamp}")
        print(f"💬 Answer Preview: {answer[:200]}...")
        print(f"📏 Answer Length: {len(answer)} characters")
        
    print(f"\n{'='*60}")
    print(f"✅ Total Q&A pairs in corpus: {len(docs_dict)}")
    print(f"{'='*60}\n")
    
    # Show file sizes
    import os
    if os.path.exists(CORPUS_FOLDER):
        faiss_file = os.path.join(CORPUS_FOLDER, "index.faiss")
        pkl_file = os.path.join(CORPUS_FOLDER, "index.pkl")
        
        if os.path.exists(faiss_file):
            size_faiss = os.path.getsize(faiss_file) / 1024  # KB
            print(f"📁 index.faiss size: {size_faiss:.2f} KB")
        
        if os.path.exists(pkl_file):
            size_pkl = os.path.getsize(pkl_file) / 1024  # KB
            print(f"📁 index.pkl size: {size_pkl:.2f} KB")

except Exception as e:
    print(f"❌ Error loading corpus: {e}")
    print("Corpus might be empty or corrupted.")


# ============================================================
# USAGE:
# 1. Save this as check_corpus.py in your project root
# 2. Run: python check_corpus.py
# 3. See all saved Q&A pairs
# ============================================================