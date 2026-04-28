\# 🛒 SmartMarket RAG Bot



Chatbot de questions-réponses sur un catalogue retail, construit avec une architecture RAG (Retrieval-Augmented Generation).



\## 🎯 Cas d'usage



Interroger un catalogue produits en langage naturel :

\- Prix et disponibilité des produits

\- Promotions en cours

\- Filtrage par prix

\- Liste des catégories



\## 🏗️ Architecture



1\. `catalogue.txt` est découpé en chunks et vectorisé via OpenAI Embeddings

2\. Les vecteurs sont stockés localement dans ChromaDB avec métadonnées (prix, catégorie)

3\. À chaque question, les chunks les plus pertinents sont récupérés

4\. Le contexte + la question sont envoyés au LLM qui génère la réponse



\## 🛠️ Stack technique



\- \*\*LangChain\*\* — orchestration du pipeline RAG

\- \*\*OpenAI\*\* — embeddings (text-embedding-3-small) + LLM (gpt-4o-mini)

\- \*\*ChromaDB\*\* — base vectorielle locale

\- \*\*Streamlit\*\* — interface utilisateur

\- \*\*Python-dotenv\*\* — gestion sécurisée des clés API



\## ⚙️ Installation



Cloner le repo et installer les dépendances :



&#x20;   git clone https://github.com/ejlee94/smartmarket-rag-bot.git

&#x20;   cd smartmarket-rag-bot

&#x20;   python -m venv venv

&#x20;   venv\\Scripts\\activate

&#x20;   pip install -r requirements.txt



Créer un fichier .env à la racine du projet :



&#x20;   OPENAI\_API\_KEY=sk-...ta-clé...



\## 🚀 Lancement



Étape 1 — Ingestion du catalogue :



&#x20;   python ingest.py



Étape 2 — Lancer l'interface :



&#x20;   streamlit run app.py



\## 💡 Fonctionnalités RAG



\- Question libre : recherche sémantique dans ChromaDB + génération de réponse via LLM

\- Filtre par prix : filtrage direct sur les métadonnées ChromaDB (ex: produits à moins de 2€)

\- Liste des catégories : extraction depuis les métadonnées, sans passer par le LLM



\## ⚠️ Limites connues



\- Le RAG n'est pas conçu pour les requêtes exhaustives sur tout le catalogue

\- Les questions numériques complexes sont gérées via métadonnées, pas via le LLM

\- Le catalogue est statique (fichier texte) — une évolution possible serait un chargement dynamique depuis une base de données

