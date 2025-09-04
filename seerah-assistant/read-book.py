import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ------------------------
# 1. Load API Key
# ------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# ------------------------
# 2. Load Documents (Quran PDF with PyMuPDF)
# ------------------------
pdf_path = "data/quran.pdf"   # replace with your own file
doc = fitz.open(pdf_path)

ayah_docs = []
for page_num, page in enumerate(doc, 1):
    text = page.get_text("text")
    ayahs = text.split("\n")  # assume each line is one ayah
    for ayah in ayahs:
        ayah_text = ayah.strip()
        if len(ayah_text) > 0:
            ayah_docs.append(
                Document(page_content=ayah_text, metadata={"page": page_num})
            )

print(f"✅ Total ayah-level docs created: {len(ayah_docs)}")

# ------------------------
# 3. Create Vector DB (FAISS) with Stronger Embeddings
# ------------------------
embeddings = OpenAIEmbeddings(
    openai_api_key=api_key,
    model="text-embedding-3-large"   # better for multilingual + semantic precision
)
vectorstore = FAISS.from_documents(ayah_docs, embeddings)

# ------------------------
# 4. LLM Setup
# ------------------------
llm = ChatOpenAI(
    openai_api_key=api_key,
    model="gpt-4o-mini",  # or "gpt-3.5-turbo"
    temperature=0
)

# ------------------------
# 5. System Prompt
# ------------------------
system_prompt = """
You are a helpful assistant that ONLY answers questions using the Quran passages provided.
If the answer is not in the Quran, you must respond strictly with:
"I don’t know. I can only answer based on the Quran."
Always cite the Surah and Ayah if available.
"""

# ------------------------
# 6. Function to Answer Queries (with Debug Mode)
# ------------------------
def answer(query: str, top_k: int = 5) -> str:
    """Return assistant response for a given query, with debug info."""
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=top_k)

    if not docs_and_scores:
        return "I don’t know. I can only answer based on the Quran."

    print("\n🔍 Retrieved passages:")
    for i, (doc, score) in enumerate(docs_and_scores, 1):
        print(f"{i}. [Score={score:.4f}] {doc.page_content[:200]}...")

    retrieved_text = "\n\n".join([doc.page_content for doc, _ in docs_and_scores])

    query_with_context = f"{system_prompt}\n\nContext from Quran:\n{retrieved_text}\n\nQuestion: {query}"
    response = llm.invoke(query_with_context)

    return response.content.strip()

# ------------------------
# 7. Interactive Chatbot Loop
# ------------------------
def chat():
    print("📖 Quran Assistant (type 'exit' to quit)\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("Assistant: Goodbye! 👋")
            break
        response = answer(query)
        print(f"Assistant: {response}\n")

if __name__ == "__main__":
    chat()