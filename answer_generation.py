from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,SystemMessage

load_dotenv()
persistent_directory = "chroma_db"

#Charger Le modele d'emmbeddings

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

#Rechercher Les k chunks les plus interessant

query = input("Posez votre question : ")

retriver = db.as_retriever(search_kwargs={"k":3})
#search_type ="similarity_score_threshold"
#search_kwargs={
#    "k": 4,
#    "score_threshold":0.3
#}

relevants_docs = retriver.invoke(query)

print(f"User Query :{query}")

print("---Context---")

for i,doc in enumerate(relevants_docs,1):
    print(f"Document{i}:;\n {doc.page_content}\n")
    
#Crée une application en faisant appel a un llm
#Combiné la question et les documents

combine_input = f"""En te basant sur les prochains documents , répond a cette question:{query}
Documents:
{chr(10).join([f"-{doc.page_content}" for doc in relevants_docs])}

S'il te plait fournis une réponse claire et utile en te basant uniquement sur ces documents.Si tu ne peux pas trouveer la réponse dans les documents , dit"Je n'ai pas assez d'information pour repondre a la question basé sur le précédant document."

"""

#Create a Genai

model = ChatGoogleGenerativeAI(
                            model="gemini-2.0-flash",
                            temperature=0.2   
)

#Definor le message pour le model
messages = [
    SystemMessage(content="Tu es un assistant très utile."),
    HumanMessage(content=combine_input)
]

#Appeler le resultat
try:
    result = model.invoke(messages)
    print(f"\n---Reponse Generee---")
    print(result.content)
except Exception as e:
    error_msg = str(e)
    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
        print("\n[ERREUR] Quota API dépassé (429 RESOURCE_EXHAUSTED).")
        print("Le quota journalier gratuit de génération de texte est épuisé.")
        print("Solutions :")
        print("  1. Attendez minuit (heure Pacifique) pour la réinitialisation du quota.")
        print("  2. Activez la facturation sur votre projet Google Cloud.")
        print("  3. Consultez https://ai.dev/rate-limit pour surveiller votre usage.")