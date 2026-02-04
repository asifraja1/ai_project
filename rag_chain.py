from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


load_dotenv()


# -------- Load DB --------
embeddings = OpenAIEmbeddings()

db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 8})


# -------- Format Docs --------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# -------- Prompt --------
prompt = ChatPromptTemplate.from_template("""
Answer ONLY from the context.
If not found say "I don't know".

Context:
{context}

Question:
{question}
""")


# -------- LLM --------
llm = ChatOpenAI(model="gpt-4o-mini")


# -------- Chain --------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)