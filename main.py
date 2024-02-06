from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
import os
import warnings
from sqlalchemy.exc import SAWarning

# Suppress specific SAWarning from SQLAlchemy about Decimal types
warnings.filterwarnings('ignore', r".*support Decimal objects natively.*", SAWarning)

# User input
userInput = "Can I get a look at all the sales records, and while you're at it, throw in the full scoop on the customers? I'm talking about getting their names, where they work, and all their contact details—address, phone number, and email. Just match up the sales to the customer IDs so I know who's who. Thanks!"

# Load environment variables from .env file
load_dotenv()

# Make sure to set your OPENAI_API_KEY in your environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found in environment variables")

db = SQLDatabase.from_uri("sqlite:///research notebook/Chinook.db")

def get_schema(_):
    return db.get_table_info()

def run_query(query):
    return db.run(query)

# Initialize the ChatOpenAI model with your API key
model = ChatOpenAI(api_key=OPENAI_API_KEY)

# Define the SQL query prompt template
sql_template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
sql_prompt = ChatPromptTemplate.from_template(sql_template)

# Define the natural language response prompt template
response_template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
response_prompt = ChatPromptTemplate.from_template(response_template)

# Build the chain for generating the SQL query
sql_response = (
    RunnablePassthrough.assign(schema=get_schema)
    | sql_prompt
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

# Invoke the SQL response chain with the question and print the SQL query
sql_query_result = sql_response.invoke({"question": userInput})
print("Generated SQL Query:")
print(sql_query_result)

# Build the full chain for generating the natural language response
full_chain = (
    RunnablePassthrough.assign(query=sql_response).assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | response_prompt
    | model
)

st.title('My Streamlit App')
st.write('This is a basic Streamlit app.')

\
# Invoke the full chain with the question and print the result
result = full_chain.invoke({"question": userInput})
print("\nNatural Language Response:")
print(result)