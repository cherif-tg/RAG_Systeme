from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
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

query = input("../")

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

model = GoogleGenerativeAI(model="")

#Definor le message pour le model
messages = [
    SystemMessage(content="Tu es un assistant très utile."),
    HumanMessage(content=combine_input)
]

#Appeler le resultat

result = model.invoke(messages)

#Afficher la reponse

print(f"\n---Reponse Generer!!---")
#Full result
#print(result)
print("Content Only:")
print(result.content)