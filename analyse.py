import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the FAISS index
print("="*80)
print("📊 ANALYZING FAISS INDEX FOR SQL METADATA")
print("="*80)

# 1. Load and inspect the FAISS index
print("\n1️⃣ LOADING FAISS INDEX...")
try:
    index = faiss.read_index("faiss_store/sql_meta.index")
    print(f"✅ Index loaded successfully!")
    print(f"   📏 Dimension: {index.d}")
    print(f"   📦 Total vectors: {index.ntotal}")
    print(f"   🔍 Index type: {type(index).__name__}")
except Exception as e:
    print(f"❌ Error loading index: {e}")
    exit(1)

# 2. Load chunks and metadata
print("\n2️⃣ LOADING CHUNKS AND METADATA...")
try:
    with open("faiss_store/sql_meta_chunks.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chunks = data["chunks"]
    metadata = data["metadata"]
    
    print(f"✅ Loaded {len(chunks)} chunks")
    print(f"✅ Loaded {len(metadata)} metadata entries")
except Exception as e:
    print(f"❌ Error loading chunks: {e}")
    exit(1)

# 3. Analyze chunk distribution
print("\n3️⃣ CHUNK DISTRIBUTION BY TYPE:")
print("-" * 80)
chunk_types = {}
for meta in metadata:
    chunk_type = meta.get('type', 'unknown')
    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

for chunk_type, count in sorted(chunk_types.items()):
    print(f"   {chunk_type:20s}: {count:3d} chunks")

# 4. Show sample chunks
print("\n4️⃣ SAMPLE CHUNKS (First 3):")
print("-" * 80)
for i in range(min(3, len(chunks))):
    print(f"\n📄 Chunk {i+1}:")
    print(f"   Type: {metadata[i].get('type')}")
    print(f"   Details: {metadata[i]}")
    print(f"   Content preview (first 300 chars):")
    print(f"   {chunks[i][:300]}...")
    print(f"   Total length: {len(chunks[i])} characters")

# 5. Analyze table-specific chunks
print("\n5️⃣ TABLE-SPECIFIC CHUNKS:")
print("-" * 80)
table_chunks = {}
for i, meta in enumerate(metadata):
    if 'table_name' in meta:
        table_name = meta['table_name']
        if table_name not in table_chunks:
            table_chunks[table_name] = []
        table_chunks[table_name].append({
            'index': i,
            'type': meta['type'],
            'size': len(chunks[i])
        })

for table_name, table_info in sorted(table_chunks.items()):
    print(f"\n   📊 {table_name}:")
    for info in table_info:
        print(f"      - Chunk {info['index']:2d}: {info['type']:20s} ({info['size']:6d} chars)")

# 6. Test similarity search
print("\n6️⃣ TESTING SIMILARITY SEARCH:")
print("-" * 80)
try:
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", local_files_only=False)
    
    # Test queries
    test_queries = [
        "What tables are available?",
        "Show me customer information",
        "What are the join conditions?",
        "Date fields and formats"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        query_embedding = model.encode([query])[0].astype('float32').reshape(1, -1)
        
        # Search top 3 results
        distances, indices = index.search(query_embedding, k=3)
        
        print(f"   Top 3 matches:")
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            print(f"      {rank}. Distance: {dist:.4f}")
            print(f"         Type: {metadata[idx].get('type')}")
            print(f"         Details: {metadata[idx]}")
            print(f"         Preview: {chunks[idx][:100].replace(chr(10), ' ')}...")

except Exception as e:
    print(f"❌ Error in similarity search: {e}")

# 7. Vector statistics
print("\n7️⃣ VECTOR STATISTICS:")
print("-" * 80)
try:
    # Reconstruct all vectors
    all_vectors = np.zeros((index.ntotal, index.d), dtype='float32')
    for i in range(index.ntotal):
        all_vectors[i] = index.reconstruct(i)
    
    print(f"   Mean vector norm: {np.mean(np.linalg.norm(all_vectors, axis=1)):.4f}")
    print(f"   Std vector norm:  {np.std(np.linalg.norm(all_vectors, axis=1)):.4f}")
    print(f"   Min vector norm:  {np.min(np.linalg.norm(all_vectors, axis=1)):.4f}")
    print(f"   Max vector norm:  {np.max(np.linalg.norm(all_vectors, axis=1)):.4f}")
except Exception as e:
    print(f"❌ Error computing statistics: {e}")

# 8. Quality assessment
print("\n8️⃣ QUALITY ASSESSMENT:")
print("-" * 80)

issues = []

# Check for empty chunks
empty_chunks = [i for i, chunk in enumerate(chunks) if len(chunk.strip()) < 50]
if empty_chunks:
    issues.append(f"⚠️  Found {len(empty_chunks)} chunks with < 50 characters")

# Check for very large chunks
large_chunks = [i for i, chunk in enumerate(chunks) if len(chunk) > 5000]
if large_chunks:
    issues.append(f"⚠️  Found {len(large_chunks)} chunks with > 5000 characters (may be too large)")

# Check metadata-chunk mismatch
if len(chunks) != len(metadata):
    issues.append(f"❌ Mismatch: {len(chunks)} chunks vs {len(metadata)} metadata entries")

# Check index-chunk mismatch
if index.ntotal != len(chunks):
    issues.append(f"❌ Mismatch: {index.ntotal} vectors vs {len(chunks)} chunks")

if issues:
    print("   Issues found:")
    for issue in issues:
        print(f"   {issue}")
else:
    print("   ✅ No issues detected!")
    print("   ✅ Your FAISS index looks healthy!")

# 9. Summary
print("\n" + "="*80)
print("📋 SUMMARY")
print("="*80)
print(f"   Total chunks indexed: {len(chunks)}")
print(f"   Vector dimension: {index.d}")
print(f"   Chunk types: {len(chunk_types)}")
print(f"   Tables covered: {len(table_chunks)}")
print(f"   Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} characters")
print("\n✅ Analysis complete!")
print("="*80)