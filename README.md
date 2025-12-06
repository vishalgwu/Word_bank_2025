# Word Bank Data Drive Hackthon 2025


# 🌍 Word Bank Data Exploration & RAG Assistant

A Retrieval-Augmented Analytics system for interactive exploration of World Bank development indicators.

---

## 📌 Overview

This project builds a complete data exploration and intelligent Q&A pipeline using **World Bank development indicators**. It combines:

* **Data Preprocessing & Cleaning**
* **Semantic Embeddings + Vector Search (Qdrant)**
* **Retrieval-Augmented Generation (RAG)**
* **LLM-based Natural-Language Question Answering**
* **Interactive Streamlit Dashboard**

The system transforms raw World Bank data into a searchable knowledge base and enables natural-language analytics on global development indicators.

---

## 🗂️ Dataset Source

All datasets come from the official **World Bank Open Data Portal**.

Source (public): [https://data.worldbank.org/](https://data.worldbank.org/)

Files used:

* `JOIN_Benchmarking_Data_2023_10_04.csv`
* `JOIN_Benchmarking_Tool_2023_10_04.xlsb`
* `cleaned_data.parquet` (processed output)

These include:

* Country-level development indicators
* Region & income classifications
* Gender and demographic metrics
* Poverty & inequality deciles
* Lending classifications and fragility indicators

---

## 🧩 Problem Statement

Large structured datasets often suffer from:

* Complex CSV/Excel navigation
* Slow filtering and validation
* Difficulty identifying insights quickly
* No semantic search or reasoning capability

### 🎯 Goal

Build an intelligent automated RAG system that can:

1. **Ingest & clean World Bank data**
2. **Store it in a vector-searchable Qdrant DB**
3. **Answer analytical questions using LLMs**
4. **Provide visual insights via Streamlit**

---

## 🎯 Project Objectives

✔️ Build a scalable **data cleaning pipeline**

✔️ Convert the dataset into a **vector-searchable knowledge base**

✔️ Implement a **Retrieval-Augmented Generation (RAG)** system for Q&A

✔️ Provide an **interactive Streamlit dashboard** for exploration

---

## 🏗️ Project Architecture

```
word_bank/
├── src/
│   ├── data/
│   │   ├── filters.py          # Data filtering utilities
│   │   ├── loader.py           # CSV/XLSB ingestion
│   │   └── preprocess.py       # Cleaning, validations, transformations
│   │
│   ├── rag/
│   │   ├── ingest.py           # Embeddings + Qdrant ingestion
│   │   ├── retrieve.py         # Vector search
│   │   ├── llm.py              # LLM interface
│   │   └── summarize.py        # Summary generation
│   │
│   ├── utils/                  # Helper utilities
│   └── viz/                    # Visualization helpers
│
├── app.py                      # Streamlit dashboard
├── chat.py                     # CLI / backend chat interface
├── cleaned_data.parquet        # Processed dataset
├── questions.txt               # Sample questions
├── req.txt                     # Dependencies
└── .env                        # API keys and configs
```

---

## 🚀 Features

### 🔍 1. Automated Data Cleaning

* Normalizes columns
* Removes duplicates
* Handles missing values
* Converts CSV/XLSB → Parquet

### 📑 2. Intelligent Vector Search (Qdrant)

* Embeds dataset rows using chosen embedding model
* Stores in local Qdrant instance
* Enables semantic similarity search

### 🤖 3. RAG Question Answering

Ask:

* "Trend of income inequality for India (2010–2020)?"
* "Which region had the highest median income in 2022?"
* "Compare fragile vs non-fragile states on gender metrics."

### 📊 4. Streamlit Dashboard

* Upload files
* Explore indicators
* Search data
* Ask LLM-based analytical questions
* Visualize insights

---

## 🔄 Project Workflow

### **Phase 1 — Data Pipeline**

* Load → Clean → Normalize → Convert to Parquet
* Apply filters & schema validation

### **Phase 2 — Vector Database**

* Run Qdrant locally
* Generate embeddings
* Store rows + metadata

### **Phase 3 — RAG System**

* Retrieve matching rows
* Rank relevance
* Forward to LLM
* Generate structured answers

### **Phase 4 — Streamlit UI**

* Visualizations
* Chat + analytics queries
* Query logs

### **Phase 5 — Deployment (Optional)**

* Docker container
* EC2 hosting
* Public dashboard

---

## 💬 Example Questions (`questions.txt`)

* "Income distribution trends by region"
* "Fragile countries in 2020 and their indicators"
* "Gender metrics summary for Sub-Saharan Africa"
* "Most common lending type for low-income countries"

---

## 🛠️ Setup Instructions

### **1. Clone the Repository**

```bash
git clone https://github.com/YOUR_USERNAME/word_bank.git
cd word_bank
```

### **2. Create Environment & Install Dependencies**

```bash
pip install -r req.txt
```

### **3. Setup Environment Variables**

Create `.env`:

```
OPENAI_API_KEY=your_key
QDRANT_URL=http://localhost:6333
DB_COLLECTION=world_bank_data
```

### **4. Run Qdrant (Docker)**

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### **5. Ingest Data into Qdrant**

```bash
python -m src.rag.ingest
```

### **6. Launch Streamlit Dashboard**

```bash
streamlit run app.py
```

---

## 🔬 Research Angle

This project supports deeper research on:

* Clustering in high-dimensional embedding space
* Semantic grouping of global development indicators
* RAG quality evaluation for structured data
* Retrieval effectiveness under different filters

---

## 📈 Future Enhancements

* Multi-agent pipeline (analysis + insight generation)
* GPT-4o/Gemini-powered advanced reasoning
* S3 integration for dynamic dataset loading
* Automated data-quality scoring
* OCR ingestion for PDFs/Excel
* Interactive maps using Plotly or PyDeck

---


