# RAG System — Retrieval-Augmented Generation

Un pipeline complet d'ingestion de documents pour un système RAG (Retrieval-Augmented Generation) construit avec **LangChain**, **Google Gemini Embeddings** et **ChromaDB**.

---

## Table des matières

- [À propos du projet](#à-propos-du-projet)
- [Architecture du système](#architecture-du-système)
- [Structure du projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Détail du pipeline](#détail-du-pipeline)
- [Gestion des quotas API](#gestion-des-quotas-api)
- [Dépendances](#dépendances)

---

## À propos du projet

Ce projet implémente la première étape d'un système RAG : le **pipeline d'ingestion**. Son rôle est de :

1. **Charger** des documents texte depuis un répertoire local
2. **Découper** ces documents en morceaux (chunks) de taille fixe
3. **Vectoriser** chaque chunk via un modèle d'embedding Google Gemini
4. **Stocker** les vecteurs dans une base de données vectorielle ChromaDB persistée sur disque

Une fois les documents indexés, un module de retrieval (non inclus ici) pourra interroger la base vectorielle pour retrouver les passages les plus pertinents à une question, et les fournir comme contexte à un LLM.

---

## Architecture du système

```
┌─────────────────────────────────────────────────────────────────┐
│                      PIPELINE D'INGESTION                       │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Fichiers│    │  Text    │    │ Gemini   │    │ ChromaDB │  │
│  │  .txt    │───▶│ Splitter │───▶│Embeddings│───▶│(on disk) │  │
│  │  (docs/) │    │ (chunks) │    │          │    │          │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                                 │
│   DirectoryLoader  CharacterText   gemini-       chroma_db/    │
│                    Splitter        embedding-001               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Structure du projet

```
RAG_systems/
│
├── ingestion_pipeline.py   # Script principal du pipeline d'ingestion
├── requirements.txt        # Dépendances Python
├── .env                    # Clé API Google (non versionné)
│
├── docs/                   # Documents sources à indexer
│   ├── google.txt
│   ├── microsoft.txt
│   ├── nvidia.txt
│   ├── spacex.txt
│   └── tesla.txt
│
└── chroma_db/              # Base vectorielle persistée (générée automatiquement)
```

---

## Prérequis

- Python **3.10+**
- Un compte [Google AI Studio](https://aistudio.google.com/) avec une clé API valide
- `pip` et `venv` (inclus avec Python)

---

## Installation

### 1. Cloner le dépôt

```bash
git clone <url-du-repo>
cd RAG_systems
```

### 2. Créer et activer l'environnement virtuel

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Configuration

Créez un fichier `.env` à la racine du projet et ajoutez votre clé API Google :

```env
GOOGLE_API_KEY=votre_clé_api_ici
```

> Pour obtenir une clé API gratuite : [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

> **Important :** Ne commitez jamais le fichier `.env` dans votre dépôt Git. Ajoutez-le à votre `.gitignore`.

---

## Utilisation

Placez vos fichiers `.txt` dans le répertoire `docs/`, puis exécutez :

```bash
python ingestion_pipeline.py
```

### Sortie attendue

```
Lecture des fichiers depuis docs...

 Document1:
 Source: docs\google.txt
 longueur du contenu: 114016 characteres
 Debut du contenu : Google LLC ...

Divisions du document en morceau avec chevauchement de 0 characteres...

--- Chunk 1 ---
Source: docs\google.txt
Longueur : 876 characteres
...

---Création de la base de données vectorielle---
Traitement du lot 1/6 (80 chunks)...
Pause de 65s pour respecter le quota de l'API (100 req/min)...
Traitement du lot 2/6 (80 chunks)...
...
--- Finished creating vector store ---
Base de données vectorielle créée et persistée dans chroma_db
```

---

## Détail du pipeline

### Étape 1 — Chargement des documents (`load_documents`)

```python
load_documents(docs_path="docs")
```

- Vérifie l'existence du répertoire `docs/`
- Charge tous les fichiers `.txt` via `DirectoryLoader` + `TextLoader`
- Encodage forcé en **UTF-8** pour éviter les erreurs de décodage sur Windows
- Affiche un aperçu des 2 premiers documents (source, longueur, début du contenu)

### Étape 2 — Découpage en chunks (`split_documents`)

```python
split_documents(documents, chunk_size=1000, chunk_overlap=0)
```

- Utilise `CharacterTextSplitter` de LangChain
- **chunk_size** : taille maximale de chaque morceau en caractères (défaut : 1000)
- **chunk_overlap** : chevauchement entre les morceaux (défaut : 0)
- Affiche un aperçu des 5 premiers chunks

| Paramètre       | Valeur par défaut | Description                                      |
|-----------------|-------------------|--------------------------------------------------|
| `chunk_size`    | 1000              | Nombre max de caractères par chunk               |
| `chunk_overlap` | 0                 | Caractères partagés entre deux chunks consécutifs |

### Étape 3 — Création du vector store (`create_vector_store`)

```python
create_vector_store(chunks, persist_directory="chroma_db", batch_size=80)
```

- Initialise le modèle d'embedding **`models/gemini-embedding-001`** via Google Generative AI
- Traite les chunks par **lots de 80** pour respecter le quota de l'API gratuite
- Crée la base ChromaDB avec une **métrique de similarité cosine**
- Persiste la base sur disque dans le dossier `chroma_db/`

---

## Gestion des quotas API

Le plan gratuit de Google Gemini est limité à **100 requêtes d'embedding par minute**. Pour éviter l'erreur `429 RESOURCE_EXHAUSTED`, le pipeline traite les chunks par lots avec une pause automatique de **65 secondes** entre chaque lot.

| Plan          | Limite                  |
|---------------|-------------------------|
| Gratuit       | 100 requêtes / minute   |
| Payant        | Selon le plan souscrit  |

> Avec ~417 chunks (5 documents), le pipeline s'exécute en **~6 lots** et prend environ **5 à 6 minutes**.

---

## Dépendances

| Bibliothèque              | Rôle                                              |
|---------------------------|---------------------------------------------------|
| `langchain`               | Framework principal pour les pipelines LLM        |
| `langchain-community`     | Loaders de documents (TextLoader, DirectoryLoader)|
| `langchain-text-splitters`| Découpage de texte en chunks                      |
| `langchain-google-genai`  | Intégration Google Generative AI (Gemini)         |
| `langchain-chroma`        | Intégration ChromaDB comme vector store           |
| `langchain-openai`        | Intégration OpenAI (pour extensions futures)      |
| `langchain-huggingface`   | Embeddings HuggingFace (pour extensions futures)  |
| `langchain-ollama`        | Modèles locaux Ollama (pour extensions futures)   |
| `sentence-transformers`   | Modèles de sentences (pour extensions futures)    |
| `python-dotenv`           | Chargement des variables d'environnement `.env`   |
