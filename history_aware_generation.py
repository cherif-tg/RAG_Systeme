from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
load_dotenv()
#Se connecter a notre base de representations vectorielle

persistent_directory = "chroma_db"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db=Chroma(persist_directory=persistent_directory,embedding_function=embeddings)

model = ChatGroq(model="llama-3.1-8b-instant",temperature=0.2)


#Enregistrer les converastions

chat_history = []

def ask_question(user_question):
    print(f"\n-- You asked: {user_question} ---")
    
    #step1: Faire comprendre la questions en utilisant l'historique
    if chat_history:
        #Demander au modele
        messages= [
            SystemMessage(content="Donne l'historique du chat, réecris la nouvelle question pour etre le premier et recherchable.Retourne juste la question réecris")
        ] + chat_history + [
            HumanMessage(content=f"New question {user_question}")
        ]
        
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Recherche de :{search_question}")
    else:
        search_question =user_question
    
    #step2: Rechercher les documents les plus relevant
    retriver = db.as_retriever(search_kwargs={"k":3})
    docs =retriver.invoke(search_question)
    
    print(f"Found {len(docs)} relevant documents:")
    for i , doc in enumerate(docs,1):
        #Afficher les deux premiere Lignes
        Lines =doc.page_content.split('\n')[:2]
        preview='\n'.join(Lines)
        print(f"  Doc {i} :{preview}...")
        
    #Step3: Le prompt final
    docs_text = "\n".join([f"- {doc.page_content}" for doc in docs])
    combined_input = f"""En te basant sur les documents suivants, Repond a cette question: {user_question}
    
    Documents:
    {docs_text}
    Donne une réponse claire et utile en utilisant unique ment les informations des documents.Si tu ne peux pas trouveer la réponse dans les documents , dit -Je n'ai pas assez d'information pour repondre a la question basé sur le précédant document-.
    """
    #step4: Recuperer la reponse
    messages= [
        SystemMessage(content="Tu es un assistant très utile qui répond au questions en te basant sur les documents fournis et l'historique des  conversations")
    ] + chat_history + [HumanMessage(content=combined_input)]
    
    result =model.invoke(messages)
    answer = result.content
    
    #Step 5 : Ajouter cette conversation a l'historique
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(SystemMessage(content=answer))
    
    print(f"Reponse : {answer}")
    
#Boucle de chat:
def start_chat():
    print("Pose moi une question press 'quit' pour quitter")
    while True:
        question =input("\nVotre question:")
        
        if question.lower() == 'quit':
            print("Au revoir")
            break
        ask_question(question)
        

if __name__ == '__main__':
    start_chat()