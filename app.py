import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import logging
import json
from datetime import datetime
import faiss
import numpy as np
import re
import requests
from sentence_transformers import SentenceTransformer
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
import sqlalchemy
from sqlalchemy import text

load_dotenv()

# ===========================
# 1. EMBEDDING MODEL FIRST
# ===========================
embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",
    local_files_only=False
)

# ===========================
# 2. CORPUS (Q&A CACHE) USING LANGCHAIN FAISS
# ===========================
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

CORPUS_FOLDER = "faiss_store/qa_corpus"
os.makedirs(CORPUS_FOLDER, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Load or create the vectorstore
if os.path.exists(CORPUS_FOLDER) and os.listdir(CORPUS_FOLDER):
    try:
        vectorstore = FAISS.load_local(
            CORPUS_FOLDER,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"Loaded corpus with {vectorstore.index.ntotal} entries")
    except Exception as e:
        print(f"Load failed ({e}) → creating new")
        vectorstore = FAISS.from_texts(["dummy"], embeddings)
        vectorstore.save_local(CORPUS_FOLDER)
else:
    vectorstore = FAISS.from_texts(["dummy"], embeddings)
    vectorstore.save_local(CORPUS_FOLDER)
    print("Created new corpus")

def search_corpus(question: str):
    """Return cached answer if similar question exists"""
    print(f"\n🔍 Searching corpus for: '{question}'")
    
    results = vectorstore.similarity_search_with_score(question, k=1) 
    ## The Above line converts the question into vectors and that k=1 means returns the top 1 result                                                                                                                                                                                                                                                                                                                                                                 
    if results and results[0][1] < 0.4: ## results[0][1] is the similarity score (lower = more similar)
        doc = results[0][0] # The document object
        similarity_score = results[0][1] # The score (0.0-1.0)
        
        print(f"✅ CORPUS HIT! (similarity score: {similarity_score:.4f})")
        print(f"📋 Returning cached answer from corpus memory")
        print("="*60)
        
        full_text = doc.page_content # "QUESTION: ...\n\nANSWER:\n..."
        if "ANSWER:\n" in full_text:
            answer = full_text.split("ANSWER:\n", 1)[1]  # Get text after "ANSWER:\n"
        else:
            answer = full_text # No delimiter, return full text
        return {"answer": answer}
    
    print(f"❌ No similar question found in corpus (threshold: 0.4)")
    print(f"   Will generate new answer...\n")
    return None


def add_to_corpus(question: str, answer: str):
    print("\n" + "="*60)
    print("🔵 USER FEEDBACK: YES - Saving to Corpus")
    print(f"Question: {question}")
    print(f"Answer length: {len(answer)} characters")
    print(f"Current corpus size BEFORE: {vectorstore.index.ntotal}")
    
    full_text = f"QUESTION: {question}\n\nANSWER:\n{answer}" #Combine question + answer into one text block
    metadata = {"question": question, "timestamp": datetime.now().isoformat()} # Create metadata metadata = {"question": question, "timestamp": "2025-01-06T14:32:45"}
     
    vectorstore.add_texts([full_text], metadatas=[metadata])
    vectorstore.save_local(CORPUS_FOLDER)
    
    print(f"✅ SUCCESSFULLY SAVED TO CORPUS!")
    print(f"📊 New corpus size: {vectorstore.index.ntotal}")
    print("="*60 + "\n")



# ===========================
# 3. ENVIRONMENT & LOGGING
# ===========================
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("logs/app.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Env variables
AZURE_ENDPOINT = os.getenv("AZURE_INFERENCE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_INFERENCE_API_KEY")
AZURE_MODEL = os.getenv("AZURE_INFERENCE_MODEL")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_SYSTEM_PROMPT = os.getenv("AZURE_SYSTEM_PROMPT")

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")

# ===========================
# 4. SQL METADATA RAG (CHUNKED)
# ===========================
import json
## ( it will load the fiass index)
faiss_index = faiss.read_index("faiss_store/sql_meta.index")
# Load chunks and metadata
with open("faiss_store/sql_meta_chunks.json", "r", encoding="utf-8") as f:
    chunk_data = json.load(f)
    chunks = chunk_data["chunks"]
    chunk_metadata = chunk_data["metadata"]

def retrieve_context(question: str, top_k: int = 3):
    
    """
    Retrieve the most relevant metadata chunks for the question.
    
    Args:
        question: User's natural language query
        top_k: Number of relevant chunks to return (default 3)
    
    Returns:
        str: Combined context from top-k most relevant chunks
    """
    # Encode the question
    ## the above line convert the question into vectors
    q_emb = embedding_model.encode([question]).astype("float32")
    
    # Search for top-k most similar chunks
    ## search ONLY in sql_meta.index
    distances, indices = faiss_index.search(q_emb, top_k)
    
    # Always include global rules (chunk 0) + retrieved chunks
    relevant_chunks = [chunks[0]]  # Global context
    
    for idx in indices[0]:
        if idx != 0:  # Don't duplicate global chunk
            relevant_chunks.append(chunks[idx])
    
    # Combine and return
    combined_context = "\n\n" + "="*60 + "\n\n".join(relevant_chunks)
    
    # Optional: Log what was retrieved (useful for debugging)
    retrieved_types = [chunk_metadata[i]['type'] for i in indices[0]]
    logger.info(f"📚 Retrieved chunks: {retrieved_types}")
    
    return combined_context 
# ===========================
# 5. DATABASE CONNECTION
# ===========================
POSTGRES_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
engine = sqlalchemy.create_engine(POSTGRES_URL)

client = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_API_KEY),
    api_version=AZURE_API_VERSION
)

SCHEMA_NAME = "stage"
TABLES = ["app_main_2024", "loan_main_2024"]

def get_table_schema(table_names: list):
    schemas = {}
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY ordinal_position;
    """
    for tbl in table_names:
        full_name = f"{SCHEMA_NAME}.{tbl}"
        try:
            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params={"schema": SCHEMA_NAME, "table": tbl})
            schemas[full_name] = df["column_name"].tolist()
        except Exception:
            schemas[full_name] = []
    return schemas

if "db_schema" not in st.session_state:
    st.session_state.db_schema = get_table_schema(TABLES)

# ===========================
# 6. CORE FUNCTIONS (unchanged)
# ===========================
def call_azure_api(messages, model=AZURE_MODEL):
    url = f"{AZURE_ENDPOINT}/chat/completions?api-version={AZURE_API_VERSION}"
    payload = {"messages": messages, "model": model, "max_tokens": 2000, "temperature": 0.0}
    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Azure API Error {response.status_code}: {response.text}")
    

def fix_common_sql_errors(sql: str) -> str:
    """
    Auto-fix common SQL generation errors before execution.
    
    Args:
        sql: Raw SQL from LLM
        
    Returns:
        Fixed SQL query
    """
    import re
    from datetime import datetime
    
    # Fix 1: CRITICAL - Replace channel_code with channel_group for DTM/DTC/FSL
    # This is the company's standard and must be enforced
    if re.search(r"channel_code\s*=\s*['\"]?(DTM|DTC|FSL)['\"]?", sql, re.IGNORECASE):
        logger.warning("🔧 CRITICAL FIX: Detected channel_code usage - replacing with channel_group ILIKE")
        
        # Replace channel_code = 'DTM' with channel_group ILIKE '%DTM%'
        sql = re.sub(
            r"(\w+\.)?channel_code\s*=\s*['\"]DTM['\"]",
            r"\1channel_group ILIKE '%DTM%'",
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            r"(\w+\.)?channel_code\s*=\s*['\"]DTC['\"]",
            r"\1channel_group ILIKE '%DTC%'",
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            r"(\w+\.)?channel_code\s*=\s*['\"]FSL['\"]",
            r"\1channel_group ILIKE '%FSL%'",
            sql,
            flags=re.IGNORECASE
        )
    
    # Fix 2: Remove undefined table aliases in single-table queries
    if sql.count("FROM stage.") == 1 and "JOIN" not in sql.upper():
        table_match = re.search(r"FROM (stage\.\w+)", sql, re.IGNORECASE)
        if table_match:
            table_name = table_match.group(1)
            alias_match = re.search(r"FROM " + re.escape(table_name) + r"\s+([a-z])\b", sql, re.IGNORECASE)
            
            if not alias_match:
                sql = re.sub(r'\b(a|app|l|loan|am|lm)\.([\w_]+)', r'\2', sql)
                logger.warning("🔧 Auto-fixed: Removed undefined table aliases")
    
    # Fix 3: Detect and warn about CURRENT_DATE usage in historical queries
    if "CURRENT_DATE" in sql.upper():
        logger.warning("⚠️  WARNING: Query uses CURRENT_DATE but database contains 2023-2024 data only!")
        logger.warning("    This query may return 0 rows. Use explicit 2024 dates instead.")
    
    # Fix 4: Replace future years with 2024
    future_years = ['2025', '2026', '2027']
    for year in future_years:
        if f"'{year}-" in sql or f'"{year}-' in sql:
            sql = re.sub(rf"['\"]({year}-\d{{2}}-\d{{2}})['\"]", 
                       lambda m: m.group(0).replace(year, '2024'), sql)
            logger.warning(f"🔧 Auto-fixed: Replaced year {year} with 2024 (data coverage: 2023-2024)")
    
    # Fix 5: Replace >= with proper symbol
    sql = sql.replace("≥", ">=").replace("≤", "<=")
    
    # Fix 6: Ensure INTERVAL uses valid units (no 'quarter')
    if "INTERVAL 'quarter'" in sql or 'INTERVAL "quarter"' in sql:
        sql = sql.replace("INTERVAL 'quarter'", "INTERVAL '3 months'")
        sql = sql.replace('INTERVAL "quarter"', "INTERVAL '3 months'")
        logger.warning("🔧 Auto-fixed: Replaced INTERVAL 'quarter' with '3 months'")
    
    # Fix 7: Detect quarter calculations with CURRENT_DATE
    if "date_trunc('quarter'" in sql.lower() and "current_date" in sql.lower():
        logger.warning("⚠️  WARNING: Quarter calculation uses CURRENT_DATE (2026) but data is from 2023-2024")
        logger.warning("    Query will return 0 rows. Use explicit Q4 2024 dates:")
        logger.warning("    applicationdate >= '2024-10-01' AND applicationdate <= '2024-12-31'")
    
    # Fix 8: Ensure semicolon at end
    sql = sql.strip()
    if not sql.endswith(";"):
        sql += ";"
    
    return sql

def enhance_question_with_context(question: str) -> str:
    """
    Enhance user question with business context to guide SQL generation.
    
    Args:
        question: Original user question
        
    Returns:
        Enhanced question with explicit instructions
    """
    enhanced = question
    question_lower = question.lower()
    
    # Detect channel mentions and add filtering hints
    if any(term in question.upper() for term in ["FSL", "FRESH START"]):
        enhanced += "\n[CONTEXT: FSL filtering requires channel_group ILIKE '%FSL%' - NEVER use channel_code]"
    
    if any(term in question.upper() for term in ["DTC", "DIRECT TO CONSUMER"]):
        enhanced += "\n[CONTEXT: DTC filtering requires channel_group ILIKE '%DTC%' - NEVER use channel_code]"
    
    if any(term in question.upper() for term in ["DTM", "DIRECT TO MERCHANT"]):
        enhanced += "\n[CONTEXT: DTM filtering requires channel_group ILIKE '%DTM%' - NEVER use channel_code]"
    
    # Handle relative date queries
    if any(phrase in question_lower for phrase in [
        'last quarter', 'previous quarter', 'recent quarter', 'Q4', 'fourth quarter'
    ]):
        enhanced += "\n[CONTEXT: 'Last quarter' = Q4 2024 (Oct-Dec 2024). Use explicit dates: '2024-10-01' to '2024-12-31']"
    
    if 'q3' in question_lower or 'third quarter' in question_lower:
        enhanced += "\n[CONTEXT: Q3 2024 = Jul-Sep 2024. Use dates: '2024-07-01' to '2024-09-30']"
    
    if 'q2' in question_lower or 'second quarter' in question_lower:
        enhanced += "\n[CONTEXT: Q2 2024 = Apr-Jun 2024. Use dates: '2024-04-01' to '2024-06-30']"
    
    if 'q1' in question_lower or 'first quarter' in question_lower:
        enhanced += "\n[CONTEXT: Q1 2024 = Jan-Mar 2024. Use dates: '2024-01-01' to '2024-03-31']"
    
    # Detect last month
    if 'last month' in question_lower or 'previous month' in question_lower:
        enhanced += "\n[CONTEXT: 'Last month' = December 2024. Use dates: '2024-12-01' to '2024-12-31']"
    
    # Detect trend/volume queries
    if any(word in question_lower for word in ['trend', 'dip', 'drop', 'increase', 'volume over', 'monthly', 'by month']):
        enhanced += "\n[CONTEXT: Trend query - GROUP BY EXTRACT(MONTH FROM applicationdate), ORDER BY month]"
    
    # Handle "total" or "count" queries
    if ('total' in question_lower or 'count' in question_lower) and any(word in question_lower for word in ['quarter', 'month', 'year']):
        enhanced += "\n[CONTEXT: Aggregation query - use explicit date range with both lower and upper bounds]"
    
    return enhanced


def generate_sql(question):
    # Enhance question with business context
    enhanced_question = enhance_question_with_context(question)
        #     # Original question
        # "Show FSL applications in August 2024"
        # # After enhancement (adds hints)
        # "Show FSL applications in August 2024
        # [CONTEXT: FSL filtering requires channel_group ILIKE '%FSL%']
        # [CONTEXT: Use dates: '2024-08-01' to '2024-08-31']"
    
    rag_context = retrieve_context(question)
       ## the above line does the search
        #**What happens:**
        # 1. Your question converts to embedding (vector)
        # 2. FAISS searches `sql_meta.index` 
        # 3. Finds top-3 most relevant chunks from YAML

        # **Example chunks retrieved:**
        # ```
        # Chunk 1: "FSL filtering: ALWAYS use channel_group ILIKE '%FSL%'"
        # Chunk 2: "Use applicationdate for app volume queries"
        # Chunk 3: "Date format: '2024-08-01' to '2024-08-31'"
    
    from datetime import datetime
    
    prompt = f"""
You are a PostgreSQL SQL generator for a lending analytics database.

=== DATA CONTEXT ===
Database Coverage: July 2023 - December 2024 (applications)
Most Complete Period: January 2024 - December 2024
Current System Date: {datetime.now().strftime('%Y-%m-%d')}

CRITICAL: This is HISTORICAL DATA. Do NOT use CURRENT_DATE in queries.

Quarter Mappings for 2024 Data:
- "last quarter" or "Q4 2024" → '2024-10-01' to '2024-12-31'
- "Q3 2024" → '2024-07-01' to '2024-09-30'
- "Q2 2024" → '2024-04-01' to '2024-06-30'
- "Q1 2024" → '2024-01-01' to '2024-03-31'

=== METADATA ===
{rag_context}

=== USER QUESTION ===
{enhanced_question}

=== CRITICAL RULES (MUST FOLLOW) ===

1. CHANNEL FILTERING (MOST IMPORTANT):
   ⚠️  ALWAYS use channel_group ILIKE for channel filtering
   ⚠️  NEVER use channel_code under ANY circumstance
   
   Required Syntax:
   - DTM: WHERE channel_group ILIKE '%DTM%'
   - DTC: WHERE channel_group ILIKE '%DTC%'
   - FSL: WHERE channel_group ILIKE '%FSL%'
   
   FORBIDDEN: channel_code = 'DTM', channel_code = 'FSL', etc.

2. DATE HANDLING:
   - Use explicit date literals: '2024-10-01'
   - NEVER use CURRENT_DATE
   - ALWAYS include both lower AND upper date bounds
   - Example: applicationdate >= '2024-10-01' AND applicationdate <= '2024-12-31'

3. SQL SYNTAX:
   - Single table: No alias prefix on columns
     Correct: WHERE channel_group ILIKE '%DTM%'
     Wrong: WHERE app.channel_group ILIKE '%DTM%'
   
   - Multi-table: Define and use aliases
     Example: FROM stage.app_main_2024 app 
              JOIN stage.loan_main_2024 loan 
              ON app.applicationid = loan.loannumber
              WHERE app.channel_group ILIKE '%FSL%'

4. USE applicationdate FOR:
   - Application volume queries
   - Submission trends
   - Monthly/quarterly aggregations

5. OUTPUT REQUIREMENTS:
   - ONLY SQL query
   - NO markdown, NO backticks, NO explanations
   - If metadata doesn't support query: return CANNOT_ANSWER

Generate the SQL query:
"""
    
    messages = [
        {"role": "system", "content": "You are a PostgreSQL expert. Always use channel_group ILIKE for channel filtering. Never use channel_code. Use explicit 2024 dates."},
        {"role": "user", "content": prompt}
    ]
    
    sql = call_azure_api(messages).strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    
    if ";" in sql:
        sql = sql.split(";")[0].strip() + ";"
    
    # Apply auto-fixes
    sql = fix_common_sql_errors(sql)
    
    return sql

def is_safe_sql(sql):
    sql_lower = sql.strip().lower()
    if not (sql_lower.startswith("select") or sql_lower.startswith("with")):
        return False
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "revoke"]
    return not any(keyword in sql_lower for keyword in forbidden)

def validate_join_rules(sql: str):
    sql_lower = sql.lower()
    has_app = "stage.app_main_2024" in sql_lower
    has_loan = "stage.loan_main_2024" in sql_lower
    if has_app and has_loan:
        if not re.search(r"\bjoin\b", sql_lower):
            raise ValueError("JOIN required when using both tables.")
        if re.search(r"customerid\s*=", sql_lower):
            raise ValueError("Invalid JOIN detected: customerid is forbidden. Use applicationid = loannumber.")
        valid_join = re.search(
            r"(?:\w+\.)?applicationid\s*=\s*(?:\w+\.)?loannumber|"
            r"(?:\w+\.)?loannumber\s*=\s*(?:\w+\.)?applicationid",
            sql_lower
        )
        if not valid_join:
            raise ValueError("Invalid JOIN detected. Required join: applicationid = loannumber")
    return sql

def run_sql(sql):
    try:
        if not is_safe_sql(sql):
            raise ValueError("Unsafe SQL detected. Only SELECT queries allowed.")
        with engine.connect() as conn:
            validated_sql = validate_join_rules(sql)
            return pd.read_sql(text(validated_sql), conn)
    except Exception as e:
        return f"SQL ERROR: {e}"

def general_chat(question):
    messages = [{"role": "system", "content": AZURE_SYSTEM_PROMPT}, {"role": "user", "content": question}]
    return call_azure_api(messages)

def summarize_result(question: str, sql: str, df: pd.DataFrame):
    preview = df.head(20).to_dict(orient="records")
    prompt = f"""
You are a senior data analyst preparing a summary for business stakeholders.

USER QUESTION:
{question}

QUERY RESULT (sample rows in JSON):
{preview}

TASK:
- Produce a BULLET-POINT summary (use "-" only, no numbering).
- Each bullet must convey a clear business insight.
- Highlight trends, dips, spikes, notable comparisons or outliers.
- If data is time-based, mention the time period.
- Be concise: 3 to 6 bullet points.
- Do NOT mention SQL, tables, or columns.
- Do NOT invent numbers.

OUTPUT FORMAT:
- Bullet point 1
- Bullet point 2
- etc.
"""
    messages = [{"role": "system", "content": "You summarize results into business bullet points."},
                {"role": "user", "content": prompt}]
    return call_azure_api(messages)

# ===========================
# 7. STREAMLIT UI
# ===========================
st.title("Jarvis ChatBot")

st.sidebar.markdown("---")
st.sidebar.caption(f"Corpus size: {vectorstore.index.ntotal} saved answers")

user_input = st.text_input("Ask your question")

if st.button("Ask"):
    if not user_input.strip():
        st.warning("Please enter a question.")
        st.stop()

    # DEBUG: Print what's happening
    print(f"\n🔵 ASK BUTTON CLICKED")
    print(f"Question: {user_input.strip()}")
    
    st.session_state.current_question = user_input.strip()
    is_db_question = any(k in user_input.lower() for k in ["table", "count", "rows", "sql", "select", "loan", "appmain"])

    # Reset from_corpus flag
    st.session_state.from_corpus = False
    st.session_state.current_answer = None
    
    # DEBUG: Print session state
    print(f"Session state reset: from_corpus={st.session_state.from_corpus}")

    if is_db_question:
        st.subheader("DB Mode")

 # Check corpus cache
        cached = search_corpus(user_input)
        if cached:
            st.info("💾 Answer retrieved from corpus memory (previous interaction)")
            st.write(cached["answer"])  
            st.session_state.current_answer = cached["answer"]
            st.session_state.from_corpus = True
        else:
            sql = generate_sql(user_input)
            st.code(sql, language="sql")

            if sql == "CANNOT_ANSWER":
                st.error("Cannot answer: required column or table does not exist.")
            elif not is_safe_sql(sql):
                st.error("Unsafe SQL detected. Only SELECT queries are allowed.")
            else:
                result = run_sql(sql)
                if isinstance(result, str):
                    st.error(result)
                else:
                    st.dataframe(result)
                    with st.spinner("Generating summary..."):
                        summary = summarize_result(user_input, sql, result)
                    st.subheader("Summary")
                    st.write(summary)
                    data_preview = result.head(10).to_string(index=False)
                    final_answer = f"**Summary:**\n{summary}\n\n**Data Preview:**\n```\n{data_preview}\n```"
                    st.session_state.current_answer = final_answer
                    
                    # DEBUG: Confirm answer stored
                    print(f"\n✅ Answer stored in session_state")
                    print(f"Answer length: {len(final_answer)} characters")
                    print(f"from_corpus flag: {st.session_state.get('from_corpus', 'NOT SET')}")

    else:
        st.subheader("General Chat Mode")
        cached = search_corpus(user_input)
        if cached:
            st.info("💾 Answer retrieved from corpus memory (previous interaction)")
            st.write(cached["answer"])
            st.session_state.current_answer = cached["answer"]
            st.session_state.from_corpus = True
        else:
            answer = general_chat(user_input)
            st.write(answer)
            st.session_state.current_answer = answer

# ========================
# FEEDBACK SECTION (Always at the end)
# ========================
print(f"\n🔍 FEEDBACK SECTION CHECK:")
print(f"  current_answer exists: {bool(st.session_state.get('current_answer'))}")
print(f"  from_corpus: {st.session_state.get('from_corpus', 'NOT SET')}")
print(f"  Should show feedback: {bool(st.session_state.get('current_answer')) and not st.session_state.get('from_corpus', False)}")

if st.session_state.get('current_answer') and not st.session_state.get('from_corpus', False):
    st.markdown("---")
    st.markdown("### Was this answer helpful?")
    
    print("✅ Feedback buttons ARE VISIBLE in UI")

    col1, col2 = st.columns(2)
    if col1.button("Yes", key="feedback_yes"):
        print("\n🔵 YES BUTTON CLICKED!")
        print(f"Calling add_to_corpus with:")
        print(f"  Question: {st.session_state.current_question}")
        print(f"  Answer length: {len(st.session_state.current_answer)}")
        
        add_to_corpus(st.session_state.current_question, st.session_state.current_answer)
        st.success("✅ Thank you! Answer saved to corpus memory.")
        st.info(f"📊 Corpus now contains {vectorstore.index.ntotal} saved Q&A pairs")
        st.balloons()
        st.rerun()

    if col2.button("No", key="feedback_no"):
        print("\n⚠️ USER FEEDBACK: NO - Answer not saved\n")
        st.info("Feedback received. Answer not saved.")
        st.rerun()
else:
    print("❌ Feedback buttons NOT VISIBLE (condition not met)")
