# seerah_gemini.py
"""
Simple RAG Assistant using LangChain + Google Gemini

Steps:
1. Create a .env file in the same folder with:
   GOOGLE_API_KEY=your_api_key_here

2. Install dependencies:
   pip install langchain langchain-google-genai langchain-community google-generativeai faiss-cpu pdfplumber python-dotenv tiktoken

3. Run:
   python seerah_gemini.py
"""

import os
from dotenv import load_dotenv
import pdfplumber

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# ------------------------------
# Load API key
# ------------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")


# ------------------------------
# 1. Load PDF text
# ------------------------------
def load_pdf_text(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text


# ------------------------------
# 2. Split into chunks
# ------------------------------
def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# ------------------------------
# 3. Build vectorstore
# ------------------------------
def build_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    return FAISS.from_texts(chunks, embeddings)


# ------------------------------
# 4. Create QA chain
# ------------------------------
def make_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro",  # ✅ full ID instead of "gemini-pro"
        temperature=0,
        google_api_key=api_key
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)



# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    pdf_file = "data/raheeq.pdf"  # change to your PDF path
    text = load_pdf_text(pdf_file)
    chunks = split_text(text)
    vectorstore = build_vectorstore(chunks)
    qa = make_qa_chain(vectorstore)

    print("\nSeerah Assistant (Gemini) ready! Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = qa.invoke({"query": query})
        print("Assistant:", answer["result"])

