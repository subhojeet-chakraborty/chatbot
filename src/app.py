import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
    You are a AI data analyst who manages a Home Inventory. You are interacting with a user who is asking you questions about the Home Inventory's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.Do not respond with 'based on your SQL query ' 
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # llm = ChatOpenAI(model="gpt-4-0125-preview")
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a AI data analyst who manages Home Inventory . You are interacting with a user who is asking you questions about the Home Inventory's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # llm = ChatOpenAI(model="gpt-4-0125-preview")
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

def handle_special_queries(user_query: str) -> str:
    greetings = {
        "hi": "Hi I am Yokie, your inventory chatbot. How can I help you?",
        "hello": "Hi I am Yokie, your inventory chatbot. How can I help you?",
        "hey": "Hi I am Yokie, your inventory chatbot. How can I help you?",
        "good morning": "Good morning!",
        "good afternoon": "Good afternoon!",
        "good night": "Good night!"
    }
    farewells = ["bye", "goodbye", "see you", "later", "quit"]
    about_bot = ["who are you", "what are you", "what do you do"]
    thanks = ["thanks", "thank you"]
    generic_queries = ["how are you", "what's up", "how's it going", "do you love me", "do you hate me", "i love you"]

    user_query_lower = user_query.lower()

    if user_query_lower in greetings:
        return greetings[user_query_lower]
    elif user_query_lower in farewells:
        return "Goodbye! Have a great day!"
    elif user_query_lower in about_bot:
        return "I am an AI assistant here to help you manage your home inventory. Ask me anything about your inventory."
    elif user_query_lower in thanks:
        return "You're welcome! If you have any more questions, feel free to ask."
    elif user_query_lower in generic_queries:
        return "As an AI assistant, I don't have feelings, but I'm here to help you!"

    # Special responses for specific queries
    if "your name" in user_query_lower:
        return "I am Yokie - your inventory chatbot."
    elif "what do you do" in user_query_lower:
        return "I am Yokie. I am here to help you with your home inventory."
    elif "guide me" in user_query_lower:
        return '''Click on the chatbox below and ask your query. 
                  For example, "tell me about the items in my inventory".'''
    elif "steps to use you" in user_query_lower:
        return "Click on the chatbox below and ask me about your home inventory."
    elif "who is your creator" in user_query_lower:
        return "I have been created by the team of HomeSync."
    elif "what can you do" in user_query_lower:
        return '''I can update the quantity of any items in your inventory, 
                  check for details of items in your inventory, 
                  and provide you with a summarized shopping list.'''
    
    # Handling "can you" queries
    if user_query_lower.startswith("can you"):
        if "home inventory" in user_query_lower:
            return "Yes- I can assist you with your home inventory."
        else:
            return "No"

    return None


def is_valid_query(user_query: str) -> bool:
    # Add more sophisticated validation as needed
    if len(user_query.split()) < 2:
        return False
    return True

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm Yokie- Your AI assistant. Ask me anything about your Home inventory."),
    ]

load_dotenv()

st.set_page_config(page_title="Home Sync", page_icon=":shark:")

st.title("Sync With Your Home")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is Our chatbot to chat with Your inventory")
    
    st.text_input("Host", value="monorail.proxy.rlwy.net", key="Host")
    st.text_input("Port", value="37821", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="ZVjUAzqrMArzyEqSCdVjrhbiiOrEcxLR", key="Password")
    st.text_input("Database", value="railway", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")
    
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    special_response = handle_special_queries(user_query)
    if special_response:
        response = special_response
    # elif not is_valid_query(user_query):
    #     response = "Sorry, I could not understand, try a different query."
    else:
        try:
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        except Exception:
            response = "Sorry, I could not understand, try a different query."
    
    st.session_state.chat_history.append(AIMessage(content=response))
    
    with st.chat_message("AI"):
        st.markdown(response)

st.markdown("""
<style>
  div[data-testid="stHorizontalBlock"] div[style*="flex-direction: column;"] div[data-testid="stVerticalBlock"] {
    border: 1px solid red;
  }
</style>
""",
  unsafe_allow_html=True,
)
