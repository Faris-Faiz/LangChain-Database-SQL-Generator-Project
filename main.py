import streamlit as st
import pandas as pd
from io import StringIO
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

# # Initialize the ChatOpenAI model with your API key
# model = ChatOpenAI(api_key=OPENAI_API_KEY)

# # Define the SQL query prompt template
# sql_template = """Based on the table schema below, write a SQL query that would answer the user's question:
# {schema}

# Question: {question}
# SQL Query:"""
# sql_prompt = ChatPromptTemplate.from_template(sql_template)

# # Define the natural language response prompt template
# response_template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
# {schema}

# Question: {question}
# SQL Query: {query}
# SQL Response: {response}"""
# response_prompt = ChatPromptTemplate.from_template(response_template)

# # Streamlit App
# st.title('SQL Query Generator')

# # Connection parameters
# db_uri = r"C:\Users\User\Documents\GitHub\LangChain-Database-SQL-Generator-Project-1\research notebook\chinook.db"  # Replace this with your database URI

# conn = st.experimental_connection(
#     'chinook.db',
#     type='sql',
#     dialect="sqlite",  # Adjust this based on the type of database you are using
#     user='',  # replace with your database username
#     password='',  # replace with your database password
#     host='',  # replace with your database host
#     port='',  # replace with your database port
#     database=''  # replace with your database name
# )

# # Insert dummy data into the SQLite database
# df_dummy.to_sql("invoices", con=conn, if_exists="replace", index=False)

# User input field
user_input = st.text_area('Enter your question:', value="", height=5)

# Button to trigger the generation
if st.button('Generate Responses'):
    try:
        # Truncate the input message to fit within the model's limit
        truncated_input = user_input[:4000]

        # # Build the chain for generating the SQL query
        # sql_response = (
        #     RunnablePassthrough.assign(schema=get_schema)
        #     | sql_prompt
        #     | model.bind(stop=["\nSQLResult:"])
        #     | StrOutputParser()
        # )

        # # Invoke the SQL response chain with the truncated question and print the SQL query
        # sql_query_result = sql_response.invoke({"question": truncated_input})

        # # Display generated SQL Query
        # st.subheader('Generated SQL Query:')
        # st.code(sql_query_result)

        # # Convert the SQL query result to a DataFrame
        # df_result = pd.read_csv(StringIO(sql_query_result), delimiter='\t', index_col=0)

        # # Display the DataFrame as a table
        # st.subheader('Natural Language Response in Table Format:')
        # st.write(df_result)

        # Generate dummy data
        dummy_data = {
            "invoiceid": range(1, 11),
            "firstname": ["John", "Jane", "Bob", "Alice", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"],
            "lastname": ["Doe", "Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore"],
            "invoicedate": pd.date_range(start="2022-01-01", periods=10, freq="D"),
            "billingaddress": ["Address" + str(i) for i in range(1, 11)],
            "billingcity": ["City" + str(i) for i in range(1, 11)],
            "billingstate": ["State" + str(i) for i in range(1, 11)],
            "billingcountry": ["Country" + str(i) for i in range(1, 11)],
            "billingpostalcode": ["PostalCode" + str(i) for i in range(1, 11)],
            "total": [round(float(i * 100), 1) for i in range(1, 11)],
        }

        df_dummy = pd.DataFrame(dummy_data)
        st.write(df_dummy)
    except Exception as e:
        # Handle exceptions gracefully
        st.error(f"An error occurred: {str(e)}")
        print(f"Exception details: {e}")

