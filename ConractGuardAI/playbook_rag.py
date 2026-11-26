from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def retrieve_playbook_rules(query: str) -> str:
    """Indexes legal_playbook.md into FAISS and retrieves the most relevant negotiation rules."""
    with open("legal_playbook.md", "r", encoding="utf-8") as f:
        text = f.read()

    sections = [s.strip() for s in text.split("## Rule ") if s.strip()]
    docs = [Document(page_content="Rule " + sec, metadata={"source": "legal_playbook.md"}) for sec in sections]

    vectorstore = FAISS.from_documents(docs, embeddings)
    matches = vectorstore.similarity_search(query, k=4)

    return "\n\n---\n\n".join([m.page_content for m in matches])
