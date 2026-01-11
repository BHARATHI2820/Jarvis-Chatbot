import faiss
import pickle
import numpy as np

def inspect_faiss_index(index_path=r'D:\AI_ChatBot\faiss_store\qa_corpus\index.faiss'):
    """Inspect FAISS index file"""
    print("=" * 50)
    print("FAISS INDEX INSPECTION")
    print("=" * 50)
    
    try:
        # Load the FAISS index
        index = faiss.read_index(index_path)
        
        print(f"\n📊 Index Statistics:")
        print(f"  - Total vectors: {index.ntotal}")
        print(f"  - Vector dimension: {index.d}")
        print(f"  - Index type: {type(index).__name__}")
        print(f"  - Is trained: {index.is_trained}")
        
        # Try to get some sample vectors if available
        if index.ntotal > 0:
            print(f"\n🔍 Sample Information:")
            print(f"  - First few vector IDs available: 0 to {min(5, index.ntotal-1)}")
            
            # Reconstruct a sample vector if the index supports it
            try:
                sample_vector = index.reconstruct(0)
                print(f"  - First vector shape: {sample_vector.shape}")
                print(f"  - First 5 values: {sample_vector[:5]}")
            except:
                print("  - Vector reconstruction not supported for this index type")
        
    except FileNotFoundError:
        print(f"❌ File not found: {index_path}")
    except Exception as e:
        print(f"❌ Error reading FAISS index: {e}")

def inspect_pickle_file(pkl_path=r'D:\AI_ChatBot\faiss_store\qa_corpus\index.pkl'):
    """Inspect pickle file contents"""
    print("\n" + "=" * 50)
    print("PICKLE FILE INSPECTION")
    print("=" * 50)
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n📦 Data Type: {type(data)}")
        
        # Inspect based on data type
        if isinstance(data, dict):
            print(f"\n🔑 Dictionary Keys: {list(data.keys())}")
            print(f"\n📝 Contents Preview:")
            for key, value in list(data.items())[:5]:  # Show first 5 items
                print(f"\n  Key: {key}")
                print(f"  Value type: {type(value)}")
                if isinstance(value, (str, int, float)):
                    print(f"  Value: {value}")
                elif isinstance(value, (list, np.ndarray)):
                    print(f"  Length: {len(value)}")
                    if len(value) > 0:
                        print(f"  First item: {value[0]}")
        
        elif isinstance(data, list):
            print(f"\n📋 List Length: {len(data)}")
            print(f"\n📝 First 3 Items:")
            for i, item in enumerate(data[:3]):
                print(f"\n  [{i}] Type: {type(item)}")
                print(f"      Value: {item}")
        
        elif isinstance(data, tuple):
            print(f"\n📦 Tuple Length: {len(data)}")
            print(f"\n📝 Contents:")
            for i, item in enumerate(data):
                print(f"\n  [{i}] Type: {type(item)}")
                if hasattr(item, '__len__') and not isinstance(item, str):
                    print(f"      Length: {len(item)}")
                
                # Special handling for docstore
                if 'docstore' in str(type(item)).lower():
                    print(f"      Docstore contains document storage")
                    try:
                        docs = item._dict if hasattr(item, '_dict') else {}
                        print(f"      Number of documents: {len(docs)}")
                        if docs:
                            print(f"\n      📄 First Document Sample:")
                            first_key = list(docs.keys())[0]
                            first_doc = docs[first_key]
                            print(f"         ID: {first_key}")
                            print(f"         Content: {str(first_doc.page_content)[:200]}...")
                            if hasattr(first_doc, 'metadata'):
                                print(f"         Metadata: {first_doc.metadata}")
                    except Exception as e:
                        print(f"      Could not inspect docstore: {e}")
                
                # Special handling for dict (usually the ID mapping)
                elif isinstance(item, dict):
                    print(f"\n      🗺️ ID Mapping Dictionary:")
                    print(f"         Total mappings: {len(item)}")
                    if item:
                        print(f"         Sample mappings:")
                        for idx, (k, v) in enumerate(list(item.items())[:3]):
                            print(f"           {k} -> {v}")
        
        else:
            print(f"\n📝 Data: {data}")
        
    except FileNotFoundError:
        print(f"❌ File not found: {pkl_path}")
    except Exception as e:
        print(f"❌ Error reading pickle file: {e}")

def check_folder_structure(folder_path=r'D:\AI_ChatBot\faiss_store\qa_corpus'):
    """Check what files exist in the folder"""
    import os
    
    print("\n" + "=" * 50)
    print("FOLDER STRUCTURE")
    print("=" * 50)
    
    if os.path.exists(folder_path):
        print(f"\n📁 Contents of '{folder_path}':")
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                print(f"  📄 {item} ({size:,} bytes)")
            else:
                print(f"  📁 {item}/")
    else:
        print(f"❌ Folder not found: {folder_path}")

# Run all inspections
if __name__ == "__main__":
    check_folder_structure()
    inspect_faiss_index()
    inspect_pickle_file()
    
    print("\n" + "=" * 50)
    print("✅ Inspection Complete")
    print("=" * 50)