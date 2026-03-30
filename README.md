# 🤖 Jarvis AI Chatbot (RAG + SQL + PDF Intelligence)

## 🚀 Overview

Jarvis is an AI-powered chatbot that combines **Retrieval-Augmented Generation (RAG)**, **SQL query generation**, and **PDF-based knowledge retrieval** into a single intelligent system.

It allows users to:

* Ask natural language questions
* Generate optimized SQL queries
* Upload PDFs and query them
* Get accurate, context-aware answers

---

## 🧠 Key Features

### 🔹 1. SQL Generation with Guardrails

* Converts natural language → SQL queries
* Uses metadata-driven RAG for accuracy
* Auto-fixes common SQL errors
* Enforces safe query execution

---

### 🔹 2. PDF RAG System

* Upload PDFs dynamically
* Extracts and chunks text
* Stores embeddings in FAISS
* Answers questions from document context

---

### 🔹 3. Vector Search (FAISS)

* Uses embedding model: `BAAI/bge-small-en-v1.5`
* Stores:

  * SQL metadata
  * PDF documents
  * Q&A memory
* Fast similarity search

---

### 🔹 4. Corpus Memory (Learning System)

* Saves previous Q&A pairs
* Reuses answers if similar question detected
* Improves performance over time

---

### 🔹 5. Streamlit UI

* Interactive chatbot interface
* Multi-tab support
* Real-time responses

---

## 🏗️ Architecture

User Query
→ Check Corpus (Memory)
→ Retrieve Context (FAISS)
→ Generate Response (Azure OpenAI)
→ Execute SQL (if needed)
→ Return Result

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Vector DB:** FAISS
* **Embeddings:** Sentence Transformers (BAAI/bge-small-en)
* **LLM:** Azure OpenAI (ChatCompletions API)
* **Database:** PostgreSQL
* **PDF Processing:** PyPDF2
* **RAG Framework:** Custom implementation

---

## 📂 Project Structure

```
├── app.py
├── create_embeddings.py
├── check_corpus.py
├── faiss_store/
├── uploaded_pdfs/
├── logs/
├── metadata/
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```
git clone https://github.com/BHARATHI2820/Jarvis-Chatbot.git
cd Jarvis-Chatbot
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Set environment variables

Create `.env` file:

```
AZURE_INFERENCE_ENDPOINT=your_endpoint
AZURE_INFERENCE_API_KEY=your_key
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
```

---

### 5. Run the app

```
streamlit run app.py
```

---

## 📸 Screenshots

> Add your UI screenshots here

---

## 🔒 Security

* Environment variables are not committed
* SQL execution is restricted to safe queries only

---

## 💡 Future Improvements

* Add authentication system
* Deploy on cloud (Azure / AWS)
* Improve UI/UX
* Add multi-language support

---

## 👨‍💻 Author

**Bharathi**

---

## ⭐ If you like this project

Give it a star on GitHub!
