import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re

load_dotenv()

st.set_page_config(
    page_title="SmartMarket Assistant",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 SmartMarket Assistant")
st.caption("Posez vos questions sur le catalogue produits et les promotions en cours.")

st.markdown("**💡 Exemples de questions :**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🏷️ Produits à moins de 2€"):
        st.session_state.exemple = "Donne moi la liste des produits ayant un prix inférieur à 2€"
with col2:
    if st.button("🥤 Rayon Boissons"):
        st.session_state.exemple = "Quels sont les produits disponibles dans le rayon Boissons ?"
with col3:
    if st.button("🎁 Promotions en cours"):
        st.session_state.exemple = "Quels produits sont en promotion en ce moment ?"
@st.cache_resource
def load_vectorstore():
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )

def load_qa_chain(vectorstore, k=10):
    prompt_template = """
Tu es un assistant pour le supermarché SmartMarket.
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
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        chain_type_kwargs={"prompt": prompt}
    )

def detecter_filtre_prix(question):
    """Détecte si la question contient un filtre de prix et retourne (operateur, valeur)"""
    q = question.lower()
    match = re.search(r"(moins|inférieur|plus|supérieur|égal)\s*(de|à|ou égal à)?\s*(\d+[\.,]?\d*)\s*€?", q)
    if match:
        operateur = match.group(1)
        valeur = float(match.group(3).replace(",", "."))
        if operateur in ["moins", "inférieur"]:
            return "lt", valeur
        elif operateur in ["plus", "supérieur"]:
            return "gt", valeur
    return None, None

def repondre(question, vectorstore):
    q_lower = question.lower()

    # Liste des catégories connues
    categories_connues = ["produits laitiers", "boissons", "boulangerie", 
                          "lessive et entretien", "epicerie salée", "épicerie salée"]

    # Détection question sur UNE catégorie spécifique → liste les produits de ce rayon
    categorie_ciblee = None
    for cat in categories_connues:
        if cat in q_lower:
            categorie_ciblee = cat
            break

    if categorie_ciblee:
        docs = vectorstore.get(include=["documents", "metadatas"])
        vus = set()
        produits = []
        for doc, meta in zip(docs["documents"], docs["metadatas"]):
            if meta.get("categorie", "").lower() == categorie_ciblee:
                nom_match = re.search(r"Nom\s*:\s*(.+)", doc)
                marque_match = re.search(r"Marque\s*:\s*(.+)", doc)
                if nom_match:
                    nom = nom_match.group(1).strip()
                    marque = marque_match.group(1).strip() if marque_match else ""
                    entree = f"{nom} — {marque}" if marque else nom
                    if entree not in vus:
                        vus.add(entree)
                        produits.append(f"- {entree}")
        if produits:
            return f"Voici les produits disponibles dans le rayon **{categorie_ciblee.title()}** :\n\n" + "\n".join(sorted(produits))
        else:
            return f"Aucun produit trouvé dans le rayon {categorie_ciblee}."

    # Détection question sur TOUTES les catégories → liste les catégories
    if any(mot in q_lower for mot in ["catégorie", "categorie", "rayons disponibles", "quels rayons"]):
        docs = vectorstore.get(include=["metadatas"])
        categories = sorted(set(
            meta["categorie"]
            for meta in docs["metadatas"]
            if meta.get("categorie")
        ))
        if categories:
            liste = "\n".join(f"- {cat}" for cat in categories)
            return f"Le catalogue contient **{len(categories)} catégories** :\n\n{liste}"
        else:
            return "Impossible de récupérer les catégories."

    # Détection filtre prix
    operateur, valeur = detecter_filtre_prix(question)
    if operateur and valeur:
        if operateur == "lt":
            filtre = {"prix": {"$lt": valeur}}
        else:
            filtre = {"prix": {"$gt": valeur}}

        docs = vectorstore.get(where=filtre, include=["documents", "metadatas"])
        vus = set()
        produits = []
        for doc, meta in zip(docs["documents"], docs["metadatas"]):
            nom_match = re.search(r"Nom\s*:\s*(.+)", doc)
            if nom_match:
                nom = nom_match.group(1).strip()
                if nom not in vus:
                    vus.add(nom)
                    prix = meta.get("prix", "?")
                    produits.append(f"- {nom} : {prix}€")
        if produits:
            seuil = f"{'moins' if operateur == 'lt' else 'plus'} de {valeur}€"
            return f"Voici les produits à {seuil} :\n\n" + "\n".join(sorted(produits))
        else:
            return "Aucun produit trouvé avec ce critère de prix."

    # Question normale → RAG classique
    qa_chain = load_qa_chain(vectorstore)
    return qa_chain.invoke(question)["result"]
vectorstore = load_vectorstore()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Gestion des exemples cliquables
if "exemple" in st.session_state and st.session_state.exemple:
    question_auto = st.session_state.exemple
    st.session_state.exemple = None
    st.session_state.messages.append({"role": "user", "content": question_auto})
    with st.chat_message("user"):
        st.markdown(question_auto)
    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            reponse = repondre(question_auto, vectorstore)
        st.markdown(reponse)
    st.session_state.messages.append({"role": "assistant", "content": reponse})

if question := st.chat_input("Ex: Quels produits sont en promotion cette semaine ?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            reponse = repondre(question, vectorstore)
        st.markdown(reponse)

    st.session_state.messages.append({"role": "assistant", "content": reponse})