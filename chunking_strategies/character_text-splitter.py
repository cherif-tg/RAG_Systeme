from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter

tesla_txt ="""
Plaintes du personnel
En 2019, l'entreprise est accusée de racisme par six de ses salariés[374]. 
Selon le New York Times, Tesla aurait proposé 100 000 $ à l'une des victimes pour éviter que l'affaire ne s'ébruite[375].

En octobre 2021, Tesla est condamné à payer 137 millions de dollars à un
ancien employé victime de racisme au sein de l'une de ses usines, en 2015-2016[376].

En décembre 2021, six employées de l'usine de Fremont portent plainte contre l'entreprise pour harcèlement sexuel[377].

En février 2022, une agence de l'État de Californie porte plainte contre Tesla pour discrimination raciale à l’encontre de travailleurs noirs dans l'usine de Fremont (Californie)[378].

En octobre 2023, la US Equal Employment Opportunity Commission (EEOC) intente un procès à Tesla pour « harcèlement raciste sévère ou généralisé » envers ses employés afro-américains[379].

En septembre 2025, un travailleur de l'usine Tesla de Fremont a poursuivi Tesla pour 51 millions de dollars après avoir été frappé et blessé par un robot de la chaîne de montage[380].

"""

sppliter1 = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=200,
    chunk_overlap=0
)

"""chunks1= sppliter1.split_text(tesla_txt)
for i,chunk in enumerate(chunks1,1):
    print(f"Chunkl {i} :{len(chunk)}")
    print(f"'{chunk}'")
    print()"""

spplitter2 = RecursiveCharacterTextSplitter(
    separators=["\n\n","\n","."," ",""],
    chunk_size=100,
    chunk_overlap=0
)

chunk2 = spplitter2.split_text(tesla_txt)
print(f"Meme text avec le RecursiveCharacterSplitter:")
for i,chunk in enumerate(chunk2,1):
    print(f"Chunkl {i} :{len(chunk)}")
    print(f"'{chunk}'")
    print()