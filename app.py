from secret_key import OPENAI_API_KEY
import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.retrievers import WikipediaRetriever

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Initialisation des modules
chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)
wikipedia_api = WikipediaAPIWrapper()
retriever = WikipediaRetriever()
theme = "langchain"                                                    # THEME A CHANGER POUR CHANGER LA PROMPT

# Récupération des documents pertinents
docs = retriever.get_relevant_documents(query=theme)

# Définition du prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Voici des informations de la page Wikipédia sur {theme}: {docs}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Invocation de ChatGPT avec un message humain
result = chat.invoke(
    [
        HumanMessage(
            content="Donne-moi des infos sur la langchain"                             # THEME A CHANGER POUR CHANGER LA PROMPT
        )
    ]
)

# Affichage des résultats
print("Docs:", docs)
print("Result:", result)