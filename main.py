from fastapi import FastAPI
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = FastAPI()

# Load CSV
data = pd.read_csv("data.csv")

# Convert to text
documents = []
for index, row in data.iterrows():
    text = f"Invoice {row['Invoice_ID']} from {row['Customer']} has amount {row['Amount']} and status {row['Status']}."
    documents.append(text)

# Create embeddings
embeddings = HuggingFaceEmbeddings()

# Create vector DB
vector_db = FAISS.from_texts(documents, embeddings)

# Create retriever
retriever = vector_db.as_retriever()

# API endpoint
@app.get("/ask")
def ask_question(query: str):
    
    query_lower = query.lower()
    relevant_docs = retriever.invoke(query)

    # 🟢 Finance Agent
    def finance_agent():
        results = []
        for doc in relevant_docs:
            if "Unpaid" in doc.page_content:
                results.append(doc.page_content)
        return {
            "agent": "finance",
            "results": results
        }

    # 🔵 Summary Agent
    def summary_agent():
        total = 0
        count = 0

        for doc in relevant_docs:
            amount = int(doc.page_content.split("amount ")[1].split()[0])
            total += amount
            count += 1

        return {
            "agent": "summary",
            "total_invoices": count,
            "total_amount": total
        }

    # 🔴 Alert Agent
    def alert_agent():
        total_unpaid = 0

        for doc in relevant_docs:
            if "Unpaid" in doc.page_content:
                amount = int(doc.page_content.split("amount ")[1].split()[0])
                total_unpaid += amount

        if total_unpaid > 10000:
            message = "High unpaid amount detected!"
        else:
            message = "No major risk"

        return {
            "agent": "alert",
            "message": message,
            "total_unpaid": total_unpaid
        }

    # 🧠 Router
    if "unpaid" in query_lower:
        return finance_agent()
    
    elif "summary" in query_lower or "total" in query_lower:
        return summary_agent()
    
    elif "risk" in query_lower or "alert" in query_lower:
        return alert_agent()
    
    else:
        return {
            "agent": "unknown",
            "message": "Query not understood"
        }