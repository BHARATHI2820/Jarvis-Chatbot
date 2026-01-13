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
    Auto-fix common SQL generation errors to match company standards
    
    New fixes added:
    1. Replace applicationdate with applicationdate
    2. Add table alias if missing
    3. Qualify columns with table alias
    """
    import re
    # ========================================
    # FIX: SQL Function Typos
    # ========================================
    sql_typos = {
        r'\bEXTRACT\s*\(\s*MNTH\b': 'EXTRACT(MONTH',  # MNTH → MONTH
        r'\bDATE_TRUNC\s*\(\s*\'mnth\'': 'DATE_TRUNC(\'month\'',
        # Add more common typos as you discover them
    }
    
    for typo_pattern, correct in sql_typos.items():
        if re.search(typo_pattern, sql, re.IGNORECASE):
            logger.warning(f"🔧 AUTO-FIX: Correcting SQL typo: {typo_pattern} → {correct}")
            sql = re.sub(typo_pattern, correct, sql, flags=re.IGNORECASE)

    # ========================================
    # NEW FIX: Column Name Hallucinations
    # ========================================
    column_hallucinations = {
        r'\bvantage_score\b': 'vantage4',
        r'\bvantagescore\b': 'vantage4', 
        r'\binterest_rate\b': 'intrate',
        r'\binterestrate\b': 'intrate',
        r'\bfico_score\b': 'fico',
    }
    for wrong, correct in column_hallucinations.items():
        if re.search(wrong, sql, re.IGNORECASE):
            logger.warning(f"🔧 AUTO-FIX: Replacing '{wrong}' with '{correct}'")
            sql = re.sub(wrong, correct, sql, flags=re.IGNORECASE)

    from datetime import datetime
    
    # ========================================
    # FIX 1: Replace applicationdate with applicationdate
    # ========================================
    if "applicationdate" in sql.lower() and "applicationdate" not in sql.lower():
        logger.warning("🔧 CRITICAL FIX: Replacing applicationdate with applicationdate (company standard)")
        
        # Replace all occurrences
        sql = re.sub(
            r'\b(app\.)?applicationdate\b',
            r'\1applicationdate',
            sql,
            flags=re.IGNORECASE
        )
    
    # ========================================
    # FIX 2: Add table alias if missing for single table queries
    # ========================================
    if "FROM stage.app_main_2024" in sql and "FROM stage.app_main_2024 app" not in sql:
        logger.warning("🔧 Adding table alias 'app' (company standard)")
        sql = sql.replace("FROM stage.app_main_2024", "FROM stage.app_main_2024 app")
        
        # Now qualify unqualified columns
        # Look for common columns that need app. prefix
        columns_to_qualify = [
            'applicationid', 'applicationdate', 'channel_group', 
            'channel_code', 'channel_name', 'customerid', 'decisionid',
            'applicationdate', 'applicationapprovaldate'
        ]
        
        for col in columns_to_qualify:
            # Match column NOT preceded by 'app.' or 'loan.'
            pattern = r'\b(?<!app\.)(?<!loan\.)(' + col + r')\b'
            replacement = r'app.\1'
            sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
    
    # ========================================
    # FIX 3: Convert static date ranges to INTERVAL format
    # ========================================
    # Look for patterns like: >= '2024-10-01' AND <= '2024-12-31'
    # Convert to: >= DATE '2024-01-01' - INTERVAL '3 months'
    
    q4_pattern = r">=\s*['\"]2024-10-01['\"].*?<=\s*['\"]2024-12-31['\"]"
    if re.search(q4_pattern, sql, re.IGNORECASE):
        logger.warning("🔧 Converting Q4 date range to INTERVAL format (company standard)")
        sql = re.sub(
            q4_pattern,
            ">= DATE '2024-01-01' - INTERVAL '3 months'",
            sql,
            flags=re.IGNORECASE
        )
    
    # ========================================
    # FIX 4: Channel filtering (keep existing logic)
    # ========================================
    if re.search(r"channel_code\s*=\s*['\"]?(DTM|DTC|FSL)['\"]?", sql, re.IGNORECASE):
        logger.warning("🔧 CRITICAL FIX: Detected channel_code usage - replacing with channel_group ILIKE")
        
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
    
    # ========================================
    # FIX 5: Detect CURRENT_DATE usage
    # ========================================
    if "CURRENT_DATE" in sql.upper():
        logger.warning("⚠️  WARNING: Query uses CURRENT_DATE but database contains 2023-2024 data only!")
        logger.warning("    This query may return 0 rows. Use explicit 2024 dates instead.")
    
    # ========================================
    # FIX 6: Replace future years with 2024
    # ========================================
    future_years = ['2025', '2026', '2027']
    for year in future_years:
        if f"'{year}-" in sql or f'"{year}-' in sql:
            sql = re.sub(rf"['\"]({year}-\d{{2}}-\d{{2}})['\"]", 
                       lambda m: m.group(0).replace(year, '2024'), sql)
            logger.warning(f"🔧 Auto-fixed: Replaced year {year} with 2024 (data coverage: 2023-2024)")
    
    # ========================================
    # FIX 7: Replace >= with proper symbol
    # ========================================
    sql = sql.replace("≥", ">=").replace("≤", "<=")
    
    # ========================================
    # FIX 8: Ensure INTERVAL uses valid units
    # ========================================
    if "INTERVAL 'quarter'" in sql or 'INTERVAL "quarter"' in sql:
        sql = sql.replace("INTERVAL 'quarter'", "INTERVAL '3 months'")
        sql = sql.replace('INTERVAL "quarter"', "INTERVAL '3 months'")
        logger.warning("🔧 Auto-fixed: Replaced INTERVAL 'quarter' with '3 months'")

    # ========================================
    # FIX 9: Ensure semicolon at end
    # ========================================
    sql = sql.strip()
    if not sql.endswith(";"):
        sql += ";"
    
    return sql


def enhance_question_with_context(question: str) -> str:
    """
    Enhance user question with business context to guide SQL generation.
    """
    enhanced = question
    question_lower = question.lower()

        # ========================================
    # TREND QUERIES = Full Year (12 months)
    # ========================================
    if 'trend' in question_lower:
        enhanced += "\n[CONTEXT: Trend query requires 12-month period - use INTERVAL '12 months']"
        enhanced += "\n[CONTEXT: Include both COUNT and AVG for comprehensive trend analysis]"
    
    # ========================================
    # LAST QUARTER = 3 months (keep existing logic)
    # ========================================
    elif any(phrase in question_lower for phrase in [
        'last quarter', 'previous quarter', 'Q4', 'quarter'
    ]):
        enhanced += "\n[CONTEXT: Last quarter = 3 months]"
    
    # Detect channel mentions and add filtering hints
    if any(term in question.upper() for term in ["FSL", "FRESH START"]):
        enhanced += "\n[CONTEXT: FSL filtering requires channel_group ILIKE '%FSL%' - NEVER use channel_code]"
    
    if any(term in question.upper() for term in ["DTC", "DIRECT TO CONSUMER"]):
        enhanced += "\n[CONTEXT: DTC filtering requires channel_group ILIKE '%DTC%' - NEVER use channel_code]"
    
    if any(term in question.upper() for term in ["DTM", "DIRECT TO MERCHANT"]):
        enhanced += "\n[CONTEXT: DTM filtering requires channel_group ILIKE '%DTM%' - NEVER use channel_code]"
    
    # Handle relative date queries
    # **CHANGED: Now uses applicationdate instead of applicationdate**
    if any(phrase in question_lower for phrase in [
        'last quarter', 'previous quarter', 'recent quarter', 'Q4', 'fourth quarter'
    ]):
        # Company uses dynamic date calculation
        enhanced += "\n[CONTEXT: 'Last quarter' = Q4 2024. Use applicationdate >= DATE '2024-01-01' - INTERVAL '3 months']"
    
    if 'q3' in question_lower or 'third quarter' in question_lower:
        enhanced += "\n[CONTEXT: Q3 2024. Use applicationdate >= DATE '2024-04-01' - INTERVAL '3 months']"
    
    if 'q2' in question_lower or 'second quarter' in question_lower:
        enhanced += "\n[CONTEXT: Q2 2024. Use applicationdate >= DATE '2024-07-01' - INTERVAL '3 months']"
    
    if 'q1' in question_lower or 'first quarter' in question_lower:
        enhanced += "\n[CONTEXT: Q1 2024. Use applicationdate >= DATE '2024-10-01' - INTERVAL '3 months']"
    
    # Detect last month
    if 'last month' in question_lower or 'previous month' in question_lower:
        enhanced += "\n[CONTEXT: 'Last month' = December 2024. Use applicationdate with INTERVAL calculation]"
    
    # **NEW: Add guidance for application count queries**
    if any(word in question_lower for word in ['app count', 'application count', 'how many app', 'total app']):
        enhanced += "\n[CONTEXT: Use applicationdate for application counts, not applicationdate]"
    
    # Detect trend/volume queries
    if any(word in question_lower for word in ['trend', 'dip', 'drop', 'increase', 'volume over', 'monthly', 'by month']):
        enhanced += "\n[CONTEXT: Trend query - GROUP BY EXTRACT(MONTH FROM applicationdate), ORDER BY month]"
    
    # Handle "total" or "count" queries
    if ('total' in question_lower or 'count' in question_lower) and any(word in question_lower for word in ['quarter', 'month', 'year']):
        enhanced += "\n[CONTEXT: Aggregation query - use applicationdate with INTERVAL calculation]"
    
    return enhanced
def generate_sql(question):
    """
    Generate SQL matching company chatbot patterns
    
    Key changes:
    1. Use applicationdate
    2. Use table aliases (app, loan)
    3. Use DATE + INTERVAL syntax
    """
    
    # Enhance question with business context
    enhanced_question = enhance_question_with_context(question)
    
    # Retrieve RAG context
    rag_context = retrieve_context(question)
    
    from datetime import datetime
    
    prompt = f"""
You are a PostgreSQL SQL generator for a lending analytics database.

=== DATA CONTEXT ===
Database Coverage: July 2023 - December 2024 (applications)
Most Complete Period: January 2024 - December 2024
Current System Date: {datetime.now().strftime('%Y-%m-%d')}

CRITICAL: This is HISTORICAL DATA. Do NOT use CURRENT_DATE in queries.
=== CRITICAL COLUMN NAME CORRECTIONS ===
NEVER use these column names (they don't exist):
❌ vantage_score
❌ vantagescore  
❌ vantage
❌ interest_rate
❌ interestrate

ALWAYS use exact column names from metadata:
✅ vantage4 (VantageScore 4.0)
✅ fico (FICO score)
✅ intrate (interest rate in loan_main_2024)
✅ apr (annual percentage rate)

=== METADATA ===
{rag_context}

=== USER QUESTION ===
{enhanced_question}

=== CRITICAL RULES ===
1. **COLUMN NAME EXACTNESS**: Only use column names EXACTLY as defined in metadata
   - For VantageScore: MUST use vantage4 (NOT vantage_score)
   - If metadata doesn't have exact column, return CANNOT_ANSWER

2. **TABLE ALIASES**: Always use table aliases
   - Single table: FROM stage.app_main_2024 app
   - Multi-table: FROM stage.app_main_2024 app JOIN stage.loan_main_2024 loan

3. **DATE FILTERING**: 
   - Last quarter = 3 months
   - Format: applicationdate >= DATE '2024-01-01' - INTERVAL '3 months'

4. **CHANNEL FILTERING**:
   - Use: channel_group ILIKE '%FSL%'
   - Never use: channel_code

=== METADATA ===
{rag_context}

=== USER QUESTION ===
{enhanced_question}

=== CRITICAL RULES (MUST FOLLOW) ===

1. **ALWAYS USE TABLE ALIASES** (COMPANY STANDARD):
   - Single table: FROM stage.app_main_2024 app
   - With alias, MUST qualify columns: app.applicationid, app.channel_group
   - Multi-table: FROM stage.app_main_2024 app JOIN stage.loan_main_2024 loan

2. **DATE COLUMN USAGE** (CRITICAL - COMPANY USES DIFFERENT COLUMN):
   ⚠️  For application counts/volume: USE applicationdate 
   ⚠️  Company standard: applicationdate tracks when borrower STARTED application
   
   Correct: WHERE app.applicationdate >= DATE '2024-01-01' - INTERVAL '3 months'
   Wrong: WHERE applicationdate >= '2024-10-01'

3. **DATE FILTERING SYNTAX** (MUST MATCH COMPANY FORMAT):
   Company uses dynamic date calculations with INTERVAL:
   
   Format: >= DATE 'YYYY-MM-DD' - INTERVAL 'N months'
   
   Examples:
   - Q4 2024 (last quarter): >= DATE '2024-01-01' - INTERVAL '3 months'
     (This calculates: 2024-01-01 minus 3 months = 2023-10-01, which is Q4 start)
   
   - Q3 2024: >= DATE '2024-07-01' - INTERVAL '3 months'
   - Q2 2024: >= DATE '2024-04-01' - INTERVAL '3 months'
   - Q1 2024: >= DATE '2024-01-01'
   
   **IMPORTANT**: The INTERVAL calculation is the company standard pattern.
   Always use: DATE 'start_date' - INTERVAL 'N months'

4. **CHANNEL FILTERING** (KEEP YOUR CURRENT APPROACH - IT'S CORRECT):
   ✅ ALWAYS use channel_group ILIKE for channel filtering
   ✅ NEVER use channel_code
   
   Required Syntax:
   - DTM: WHERE app.channel_group ILIKE '%DTM%'
   - DTC: WHERE app.channel_group ILIKE '%DTC%'
   - FSL: WHERE app.channel_group ILIKE '%FSL%'

5. **SQL STRUCTURE** (MATCH COMPANY FORMAT):
   
   Template for single table query:
   ```sql
   SELECT COUNT(app.applicationid) AS total_app_count
   FROM stage.app_main_2024 app
   WHERE app.applicationdate >= DATE '2024-01-01' - INTERVAL '3 months'
     AND app.channel_group ILIKE '%DTM%';
   ```
   
   Template for multi-table query:
   ```sql
   SELECT COUNT(app.applicationid) AS total_count
   FROM stage.app_main_2024 app
   JOIN stage.loan_main_2024 loan ON app.applicationid = loan.loannumber
   WHERE app.applicationdate >= DATE '2024-01-01' - INTERVAL '3 months'
     AND app.channel_group ILIKE '%FSL%';
   ```

6. **COLUMN QUALIFICATION** (COMPANY STANDARD):
   - ALWAYS qualify columns with table alias when alias is defined
   - Correct: app.applicationdate, app.channel_group, app.applicationid
   - Wrong: applicationdate, channel_group (without alias)

7. **OUTPUT REQUIREMENTS**:
   - ONLY SQL query
   - NO markdown, NO backticks, NO explanations
   - If metadata doesn't support query: return CANNOT_ANSWER
   - End with semicolon

=== EXAMPLES (COMPANY FORMAT) ===

Q: "What is total app count for DTM during the last quarter"
A: SELECT COUNT(app.applicationid) AS total_app_count FROM stage.app_main_2024 app WHERE app.applicationdate >= DATE '2024-01-01' - INTERVAL '3 months' AND app.channel_group ILIKE '%DTM%';

Q: "Show FSL applications in August 2024"
A: SELECT COUNT(app.applicationid) AS total_count FROM stage.app_main_2024 app WHERE app.applicationdate >= DATE '2024-08-01' AND app.applicationdate < DATE '2024-09-01' AND app.channel_group ILIKE '%FSL%';

Q: "Count DTM apps by month in Q4"
A: SELECT EXTRACT(MONTH FROM app.applicationdate) AS month, COUNT(app.applicationid) AS app_count FROM stage.app_main_2024 app WHERE app.applicationddate >= DATE '2024-01-01' - INTERVAL '3 months' AND app.channel_group ILIKE '%DTM%' GROUP BY EXTRACT(MONTH FROM app.applicationdate) ORDER BY month;

Q: "What is the vantage trend for DTM apps"
A: SELECT EXTRACT(MONTH FROM app.applicationdate) AS application_month, COUNT(app.applicationid) AS total_app_count, AVG(app.vantage4) AS avg_vantage_score FROM stage.app_main_2024 app WHERE app.applicationdate >= DATE '2024-01-01' - INTERVAL '12 months' AND app.channel_group ILIKE '%DTM%' GROUP BY EXTRACT(MONTH FROM app.applicationdate) ORDER BY application_month;

Q: "Show me FICO trend for FSL loans over the year"  
A: SELECT EXTRACT(MONTH FROM app.applicationdate) AS application_month, COUNT(app.applicationid) AS total_app_count, AVG(app.fico) AS avg_fico_score FROM stage.app_main_2024 app WHERE app.applicationdate >= DATE '2024-01-01' - INTERVAL '12 months' AND app.channel_group ILIKE '%FSL%' GROUP BY EXTRACT(MONTH FROM app.applicationdate) ORDER BY application_month;

Generate the SQL query following these exact patterns:
"""
    
    messages = [
        {"role": "system", "content": "You are a PostgreSQL expert. Generate SQL matching the company's exact patterns: use applicationdate, table aliases, and DATE + INTERVAL syntax."},
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

def validate_column_names(sql: str) -> str:
    """
    Validate and fix common column name hallucinations
    """
    import re
    
    # Define column name mappings (wrong → correct)
    column_fixes = {
        r'\bvantage_score\b': 'vantage4',
        r'\bvantagescore\b': 'vantage4',
        r'\bvantage\b(?!4)': 'vantage4',  # vantage but not vantage4
        r'\binterest_rate\b': 'intrate',
        r'\binterestrate\b': 'intrate',
        r'\bfico_score\b': 'fico',
    }
    
    original_sql = sql
    
    for wrong_pattern, correct_name in column_fixes.items():
        if re.search(wrong_pattern, sql, re.IGNORECASE):
            logger.warning(f"🔧 CRITICAL FIX: Replacing hallucinated column '{wrong_pattern}' with '{correct_name}'")
            sql = re.sub(wrong_pattern, correct_name, sql, flags=re.IGNORECASE)
    
    if sql != original_sql:
        logger.info("✅ Column name validation applied")
    
    return sql


def run_sql(sql):
    try:
        if not is_safe_sql(sql):
            raise ValueError("Unsafe SQL detected. Only SELECT queries allowed.")
        
        # Add validation step
        sql = validate_column_names(sql)  # ← ADD THIS LINE
        
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
