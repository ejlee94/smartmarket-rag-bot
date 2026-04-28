from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import re, os, shutil

load_dotenv()

print("📄 Chargement du catalogue...")
loader = TextLoader("catalogue.txt", encoding="utf-8")
documents = loader.load()

print("✂️  Découpage en chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# Extraction du prix et de la catégorie pour chaque chunk
for chunk in chunks:
    text = chunk.page_content

    prix_match = re.search(r"Prix\s*:\s*([\d.]+)€", text)
    chunk.metadata["prix"] = float(prix_match.group(1)) if prix_match else None

    cat_match = re.search(r"Catégorie\s*:\s*(.+)", text)
    chunk.metadata["categorie"] = cat_match.group(1).strip() if cat_match else None

print(f"   → {len(chunks)} chunks créés avec métadonnées")

# Supprime l'ancienne base et recrée
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

print("🧠 Création des embeddings et stockage ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db"
)

print("✅ Ingestion terminée avec métadonnées !")