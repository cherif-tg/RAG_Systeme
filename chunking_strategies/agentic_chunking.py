from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0.2)

spacex_text ="""Le site de lancement à Cap Canaveral.
Le siège, les bureaux d'études et les installations industrielles de SpaceX sont situés à Hawthorne en Californie, près de l'aéroport de Los Angeles. SpaceX y dispose d'une surface couverte de 5,1 hectares permettant d'assembler en parallèle trois lanceurs Falcon 9 ainsi que deux douzaines de moteurs Merlin et trois lanceurs Falcon 1[62].

Les moteurs sont testés sur un banc d'essais situé à McGregor au Texas. Ce site est aussi utilisé pour les tests du prototype « Grasshopper », consistant à faire décoller et atterrir verticalement le premier étage d'une fusée Falcon 9[63].

Pour lancer ses Falcon 9, ses Falcon Heavy et ses Starship, la société dispose :

du pad SLC-40 : une installation de lancement sur la base de Cap Canaveral ;
du pad LC-39A : au centre spatial Kennedy (le complexe de lancement 39A), réaménagement d'un pas de tir du centre spatial Kennedy utilisé autrefois pour le lancement de la Saturn V ou de la Navette spatiale américaine afin d'y effectuer les tirs du lanceur Falcon 9 ou Falcon Heavy ;
du pad SLC-4E à Vandenberg Air Force Base pour les tirs depuis la côte ouest des États-Unis ;
une quatrième base de lancement, réservée aux Starships, la SpaceX Starbase est construite à Boca Chica Village à environ 25 km à l'est de Brownsville (État du Texas) en bordure du golfe du Mexique et à quelques kilomètres de la frontière entre les États-Unis et le Mexique. Contrairement aux autres installations de lancement qui dépendent du gouvernement américain (NASA et l’Armée de l'Air américaine), la Starbase appartient en propre à la société, ce qui lui donne plus de latitude dans l'exploitation du site. SpaceX a investi 100 millions de dollars (environ 88 millions d'euros) dans ce complexe de lancement. Les travaux ont débuté en 2015[64] et le premier lancement (suborbital) depuis ce site a lieu en 2019.
Pour l'atterrissage des premiers étages des lanceurs Falcon 9 ou Heavy, la société dispose :

des Landing Zone 1 (LZ-1) et LZ-2, situées à Cap Canaveral, utilisables lors des lancements depuis le pad SLC-40 et ceux depuis le pad LC-39A ;
de la Landing Zone 4 (LZ-4) à Vandenberg Air Force Base (proche du pad SLC-4E), lors de lancements depuis la côte Est.
de deux barges de récupération, « Of Course I Still Love You » sur la Côte-Est et « Just Read the Instructions » sur la Côte-Ouest.
L'entreprise emploie en tout environ 9 500 personnes.

Courant juillet 2024, Elon Musk annonce le déménagement du siège social de SpaceX à partir de Hawthorne jusqu'à la Starbase au Texas, le justifiant par la mise en place d'une loi de l'État de Californie sur la question de la transidentité[65], mais en pratique aussi pour des raisons fiscales et réglementaires[66].

Les productions de la société SpaceX
Le lanceur super-lourd Starship
Article détaillé : Starship (fusée).
Starship est la fusée créée par SpaceX pour permettre la colonisation de la Lune ou Mars, développer le tourisme spatial, et relier plusieurs points de la Terre en moins de 50 minutes (Paris-New York en 30 minutes). Il agit en tandem avec le booster SuperHeavy, qui sera réutilisable, comme un booster de Falcon 9.

Ce vaisseau serait capable de transporter 100 tonnes en orbite basse, puis 150 tonnes lorsque optimisé.

Les premiers essais du prototype StarHopper se sont déroulés en avril 2019 depuis Boca Chica Village au sud du Texas, où le véhicule a réalisé ses premiers sauts. Ce prototype est destiné aux toutes premières phases de test du véhicule, c'est-à-dire plusieurs décollages progressifs, jusqu'aux tests orbitaux.
"""

prompt =f"""
Tu es un expert en chunking. Divise Ce text en chunks logiques.
Regles:
1. Chaque chunk doit etre autour de 200 caracteres ou moins
2.Divise en sujets naturels
3.ajoute <<SPLIT>> entre chaque chunk

text:
{spacex_text}

Retourne le text avec <<SPLIT>>  comme markeur la ou tu veux diviser
"""

print("Damande a l'IA de Diviser le text...")
reponse= llm.invoke(prompt)
marked_text =reponse.content


#Division 
chunks= marked_text.split("<<SPLIT>>")

clean_chunk =[]
for chunk in chunks:
    cleaned =chunk.strip()
    if cleaned:
        clean_chunk.append(cleaned)
        
print("\n Agentic reponse")
print("=" *50)

for i ,chunk in enumerate(clean_chunk,1):
    print(f"chunk {i} :({len(chunk)} caracters)")
    print(f'"{chunk}')
    print()