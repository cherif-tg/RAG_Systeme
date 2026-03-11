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
    
