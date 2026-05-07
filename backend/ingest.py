from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def ingest_documents() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    documents_dir = repo_root / "data" / "documents"
    faiss_dir = repo_root / "backend" / "faiss_index"

    pdf_paths = sorted(documents_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {documents_dir}")

    docs = []
    total_pdfs = len(pdf_paths)
    for idx, pdf_path in enumerate(pdf_paths, start=1):
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
        if idx % 10 == 0 or idx == total_pdfs:
            print(f"Processed {idx}/{total_pdfs} documents")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    faiss_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(faiss_dir))

    print(f"Indexed {len(chunks)} total chunks into FAISS at {faiss_dir}")


if __name__ == "__main__":
    try:
        ingest_documents()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully.")
