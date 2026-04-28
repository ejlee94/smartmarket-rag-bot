from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 1. Connexion à la base vectorielle existante
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
)

# 2. Prompt personnalisé — contexte retail
prompt_template = """
Tu es un assistant pour le supermarché Fraîcheur+.
Réponds uniquement à partir des informations du catalogue ci-dessous.
Si tu ne trouves pas l'information, dis-le clairement.
Lorsque tu listes des produits, précise toujours que c'est une sélection d'exemples et non une liste exhaustive.

Contexte :
{context}

Question : {question}

Réponse :"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 3. Chaîne RAG
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
    chain_type_kwargs={"prompt": prompt}
)

# 4. Test rapide
if __name__ == "__main__":
    questions = [
        "Quels produits laitiers sont en promotion ?",
        "Quel est le prix du Camembert de Normandie ?",
        "Quelles lessives sont disponibles ?"
    ]
    for q in questions:
        print(f"\n❓ {q}")
        print(f"💬 {qa_chain.invoke(q)['result']}")