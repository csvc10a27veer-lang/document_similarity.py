Semantic Document Similarity with LangChain

An intelligent text-matching application that utilizes sentence embeddings to calculate semantic similarity between a user query and a document corpus. Unlike traditional keyword matching, this system leverages vector spaces to understand the contextual meaning behind sentences.

## 🚀 Features
* **Dual Pipeline Support:** Easily switch between local execution (Hugging Face) and cloud APIs (OpenAI).
* **Dimensionality Tuning:** Configured to handle optimized vector lengths.
* **Vector Math Evaluation:** Uses Cosine Similarity to calculate the mathematical distance between query and context arrays.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.10+
* **Framework:** [LangChain](https://github.com/langchain-ai/langchain)
* **ML/Math Libraries:** Scikit-Learn, NumPy
* **Embedding Models:** * Local: `sentence-transformers/all-MiniLM-L6-v2` (via Hugging Face)
  * Cloud: `text-embedding-3-large` (via OpenAI)
