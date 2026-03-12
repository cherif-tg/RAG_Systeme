import os
import time
from langchain_community.document_loaders import TextLoader, DirectoryLoader,PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
"""from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # léger & efficace
    # ou "BAAI/bge-m3" pour le multilingue (français inclus)
)"""

def load_pdf_documents(docs_path="docs_pdf"):
    """Charge des documents pdf a partir du répertoire spécifié"""
    print(f"Lecture des documents de {docs_path}")
    #On verifie si le document existe
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Le repertoire spécifié n'existe pas")
    loader =DirectoryLoader(
        path=docs_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        loader_kwargs={"encoding":"utf-8"}
    )
    documents=loader.load()
    if len(documents) == 0:
        raise FileNotFoundError(f"Le repertoire est vide")
    
    for i, doc in enumerate(documents[:4],1):
        print(f"\nDocument{i}:")
        print(f"Source:{doc.metadata['source']}")
        print(f"Longueur du contenu :{len(doc.page_content)} caracteres")
        print(f"Debut du document :{doc.page_content[:100]}")
        print(f"metadata:{doc.metadata}")
    return documents
    

def load_documents(docs_path = "docs"):
    """Charge les documents txt à partir du répertoire spécifié et affiche des informations sur les deux premiers documents."""
    #Verifier si docs existe
    print(f"Lecture des fichiers depuis {docs_path}...")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Le repertoire {docs_path} n'existe pas crée le et ajouter vos fichiers.")

    #Lire tout les fichiers txt
    loader =DirectoryLoader(
        path = docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents =loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"Pas de fichiers txt dans le repertoire{docs_path}")
    
    for i , doc in enumerate(documents[:2]):
        print(f"\n Document{i+1}:")
        print(f" Source:{doc.metadata['source']}")
        print(f" longueur du contenu:{len(doc.page_content)} characteres")                           
        print(f" Debut du contenu :{doc.page_content[:100]}...")
        print(f" metadata: {doc.metadata} ")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Divise les documents en morceaux de taille spécifiee et affiche des informations sur les deux premiers morceaux."""
    print(f"Divisions du document en morceau avec chevauchement de {chunk_overlap} characteres...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    """Autres methodes de chunking:
    -CharacterTextSplitter -> Divise les texte par separateurs( par defaut \n\n)
    1. RecursiveCharacterTextSpliter(Version ameeliorer de CharacterTextSpliter) -> Divise les sequences naturelles commes(phrases, paragraphes,mots)
        -Preserve le context des xhunks
    2.Document-Specific Splitting(Respecte la structure des documents)
        -pdf: diviser par pages, sections
        -Chaque type de document a son traitement approprié
    3.La Division Semantic(Semantic spliting)
        -Utilise les embeddings pour detecter les meilleurs shifts
        -Plus intelligent mais plus lourd
    4. LA Division Agentic(Division propulser par l'IA)
        -Le model analyse lui meme le contexte et decide des meilleurs divisions
        -Est performant sur les relations complexe
    """
    chunks=text_splitter.split_documents(documents)
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Longueur : {len(chunk.page_content)} characteres")
            print(f"Content:")
            print(chunk.page_content[:200] + "...")
            print("-" *50)
    
    if len(chunks) > 5:
        print(f"\n... and {len(chunks)- 5} more chunks")
        return chunks
    
def create_vector_store(chunks, persist_directory="chroma_db", batch_size=80):
    """Crée une base de données vectorielle à partir des morceaux de documents"""

    print("---Création de la base de données vectorielle---") 
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    #Divisions des chunks en lot de 80 pour ne pas depasser de l'api gemini gratuit
    total_batches = (len(chunks) - 1) // batch_size + 1
    vector_store = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"Traitement du lot {batch_num}/{total_batches} ({len(batch)} chunks)...")

        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings_model,
                persist_directory=persist_directory,
                collection_metadata={"hnsw:space": "cosine"}
            )
        else:
            vector_store.add_documents(batch)

        if i + batch_size < len(chunks):
            print("Pause de 65s pour respecter le quota de l'API (100 req/min)...")
            time.sleep(65)

    print("--- Finished creating vector store ---")
    print(f"Base de données vectorielle créée et persistée dans {persist_directory}")
    return vector_store


def main():
    #1.Loading the files
    documents = load_documents(docs_path="docs")
    
    #2.Diviser les documents en morceaux
    chunks = split_documents(documents)
      
    #3.Créer la base de données vectorielle
    vector_store  =create_vector_store(chunks)

if __name__ == "__main__":
    main()