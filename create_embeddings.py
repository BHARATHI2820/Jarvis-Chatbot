import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re
import json

# Load embedding model
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5", local_files_only=False)

# Load YAML as plain text (bypass YAML parsing)
yaml_path = "metadata/meta.yaml"

with open(yaml_path, "r", encoding="utf-8") as f:
    yaml_text = f.read()

print("✅ Loaded meta.yaml as text")

# ==========================
# INTELLIGENT TEXT-BASED CHUNKING
# ==========================

chunks = []
metadata = []

# 1. Extract global section (first 100 lines)
lines = yaml_text.split("\n")
global_section = "\n".join(lines[:100])
chunks.append(global_section)
metadata.append({"type": "global", "section": "header_and_schema"})

# 2. Extract each table section using regex
table_pattern = r"(stage\.\w+):"
table_matches = list(re.finditer(table_pattern, yaml_text))

for i, match in enumerate(table_matches):
    table_name = match.group(1)
    start_pos = match.start()
    
    # Find end position (next table or end of file)
    if i < len(table_matches) - 1:
        end_pos = table_matches[i + 1].start()
    else:
        end_pos = len(yaml_text)
    
    # Extract table content
    table_content = yaml_text[start_pos:end_pos]
    
    # Further split large tables
    table_lines = table_content.split("\n")
    
    # Split into columns section and rules section
    columns_end = None
    rules_start = None
    
    for idx, line in enumerate(table_lines):
        if "rules:" in line:
            rules_start = idx
            columns_end = idx
            break
    
    if columns_end and rules_start:
        # Columns chunk
        columns_chunk = "\n".join(table_lines[:columns_end])
        chunks.append(columns_chunk)
        metadata.append({"type": "table_columns", "table_name": table_name})
        
        # Rules chunk
        rules_chunk = "\n".join(table_lines[rules_start:])
        chunks.append(rules_chunk)
        metadata.append({"type": "table_rules", "table_name": table_name})
    else:
        # Single chunk for entire table
        chunks.append(table_content)
        metadata.append({"type": "table", "table_name": table_name})

# 3. Extract relationships section
relationships_match = re.search(r"relationships:(.*?)(?=glossary:|notes:|$)", yaml_text, re.DOTALL)
if relationships_match:
    rel_content = "relationships:" + relationships_match.group(1)
    chunks.append(rel_content)
    metadata.append({"type": "relationships", "section": "join_rules"})

# 4. Extract glossary
glossary_match = re.search(r"glossary:(.*?)(?=notes:|date_fields:|rules:|$)", yaml_text, re.DOTALL)
if glossary_match:
    glossary_content = "glossary:" + glossary_match.group(1)
    chunks.append(glossary_content)
    metadata.append({"type": "glossary", "section": "definitions"})

# 5. Extract date_fields section
date_fields_match = re.search(r"date_fields:(.*?)(?=rules:|$)", yaml_text, re.DOTALL)
if date_fields_match:
    date_content = "date_fields:" + date_fields_match.group(1)
    chunks.append(date_content)
    metadata.append({"type": "date_casting", "section": "date_fields"})

print(f"✅ Created {len(chunks)} chunks from metadata")
print(f"Chunk breakdown:")
for m in metadata:
    print(f"  - {m['type']}: {m.get('table_name', m.get('section', 'N/A'))}")

# Create embeddings for all chunks
print("\n🔄 Generating embeddings...")
## this below line will convert the chunks  into vectors 
embeddings = embedding_model.encode(chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32") 


# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and chunks
os.makedirs("faiss_store", exist_ok=True)
faiss.write_index(index, "faiss_store/sql_meta.index")

# Save chunks and metadata for retrieval
with open("faiss_store/sql_meta_chunks.json", "w", encoding="utf-8") as f:
    json.dump({
        "chunks": chunks,
        "metadata": metadata
    }, f, indent=2)

print(f"\n✅ FAISS index created with {len(chunks)} chunks")
print(f"📁 Saved to: faiss_store/sql_meta.index")
print(f"📁 Saved to: faiss_store/sql_meta_chunks.json")
print(f"\n🎉 Success! Now run: streamlit run app.py")