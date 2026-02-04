
import os
import re
from dotenv import load_dotenv

from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma



load_dotenv()

os.environ["USER_AGENT"] = "rag-bot"



URL = input("Enter Wikipedia URL: ")





loader = WebBaseLoader(URL)

documents = loader.load()



if documents:
    print("\n--- RAW CONTENT SAMPLE ---\n")
    print(documents[0].page_content[:2000])
else:
    print("❌ No content loaded")
    exit()



def clean_html(raw_html):

    soup = BeautifulSoup(raw_html, "html.parser")

    
    for tag in soup([
        "nav",
        "footer",
        "aside",
        "script",
        "style",
        "table"
    ]):
        tag.decompose()

    text = soup.get_text(separator=" ")

   
    text = re.sub(r"\[\d+\]", "", text)

    
    text = re.sub(r"\s+", " ", text)

    
    text = re.split(r"References", text)[0]

    return text.strip()



for doc in documents:
    doc.page_content = clean_html(
        doc.page_content
    )



splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)

print(f"\n✅ Total chunks created: {len(docs)}")


embeddings = OpenAIEmbeddings()



db = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="db"
)

db.persist()

print("\n🎉 Ingestion Complete — DB Ready")