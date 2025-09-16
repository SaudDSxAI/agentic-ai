import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document

# ------------------------
# 1. Load API Key
# ------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# ------------------------
# 2. Settings
# ------------------------
pdf_path = "data/quran.pdf"
index_path = "quran_faiss_index"

# ------------------------
# 3. Load or Build Vectorstore
# ------------------------
embeddings = OpenAIEmbeddings(
    openai_api_key=api_key,
    model="text-embedding-3-small"   # faster + cheaper
)

if os.path.exists(index_path):
    print("📂 Loading existing FAISS index...")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    # Rebuild ayah_docs for BM25
    ayah_docs = list(vectorstore.docstore._dict.values())
else:
    print("📖 Reading Quran PDF with PyMuPDF...")
    doc = fitz.open(pdf_path)
    ayah_docs = []

    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text")
        ayahs = text.split("\n")
        for ayah in ayahs:
            ayah_text = ayah.strip()
            if len(ayah_text) > 0:
                ayah_docs.append(
                    Document(page_content=ayah_text, metadata={"page": page_num})
                )

    print(f"✅ Total ayah-level docs created: {len(ayah_docs)}")

    # Build FAISS
    print("⚡ Creating FAISS index...")
    vectorstore = FAISS.from_documents(ayah_docs, embeddings)
    vectorstore.save_local(index_path)
    print("💾 FAISS index saved!")

# ------------------------
# 4. Hybrid Retriever (BM25 + FAISS)
# ------------------------
bm25_retriever = BM25Retriever.from_documents(ayah_docs)
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Weighted combo: 70% semantic (FAISS), 30% keyword (BM25)
hybrid_retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

# ------------------------
# 5. LLM Setup
# ------------------------
llm = ChatOpenAI(
    openai_api_key=api_key,
    model="gpt-4o-mini",
    temperature=0
)

# ------------------------
# 6. System Prompt
# ------------------------
system_prompt = """
You are a deeply knowledgeable assistant specializing in the Quran. Your task is to explore verses at the word-by-word level, uncovering the full depth of meaning and connecting these meanings to wider fields of knowledge.

Method

Word Breakdown

Isolate each word of the verse in Arabic.

Provide its root letters and morphology.

Semantic Range

List all possible meanings from classical Arabic lexicons.

Include literal, metaphorical, and contextual usages.

Quranic Context

Show how the word (or its root) appears in other parts of the Quran.

Highlight recurring themes and connections.

Cross-Disciplinary Reflections

For each meaning, explore possible insights across domains such as:

Natural Sciences (biology, physics, cosmology, earth sciences)

Philosophy & Metaphysics (existence, causality, consciousness, the unseen)

Psychology & Anthropology (mind, behavior, human origins, societies)

History & Civilization Studies (rise and fall of nations, knowledge transmission)

Future Studies (AI, space exploration, sustainability, ethics of technology)

Integration with Quranic Themes

Relate insights back to Quranic motifs such as reflection, creation, human responsibility, knowledge, and the unseen.

Clarity of Levels

Always distinguish between:

Explicit Quranic meaning.

Linguistically possible interpretations.

Speculative or imaginative reflections.

Style & Tone

Respectful, reflective, and imaginative.

Encourage curiosity and wonder rather than closure.

Weave Quranic wisdom with the richness of human knowledge.
"""

# ------------------------
# 7. Answer Function
# ------------------------h
def answer(query: str, top_k: int = 5) -> str:
    docs = hybrid_retriever.get_relevant_documents(query)

    if not docs:
        return "I don’t know. I can only answer based on the Quran."

    print("\n🔍 Retrieved passages:")
    for i, doc in enumerate(docs[:top_k], 1):
        print(f"{i}. {doc.page_content[:200]}...")

    retrieved_text = "\n\n".join([doc.page_content for doc in docs[:top_k]])
    query_with_context = f"{system_prompt}\n\nContext from Quran:\n{retrieved_text}\n\nQuestion: {query}"
    response = llm.invoke(query_with_context)

    return response.content.strip()

# ------------------------
# 8. Chatbot Loop
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