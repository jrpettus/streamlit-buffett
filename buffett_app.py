import snowflake.connector
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

from streamlit_chat import message

from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.llm_math.base import LLMMathChain
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool, load_tools
from langchain import SQLDatabase, SQLDatabaseChain
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
from langchain.chains import ChatVectorDBChain # for chatting with the pdf
from langchain.vectorstores import Chroma, Pinecone # for the vectorization part

import pinecone

from sqlalchemy import create_engine
from sqlalchemy.dialects import registry
registry.register('snowflake', 'snowflake.sqlalchemy', 'dialect')

sf_user = st.secrets["user"]
sf_pw = st.secrets["password"]
sf_acct = st.secrets["account"]
sf_db = st.secrets["database"]
sf_schema = st.secrets["schema"]
sf_wh = st.secrets["warehouse"]

st.set_page_config(layout="wide")

# Uses st.cache_resource to only run once.
@st.cache_resource
def llm_connection(temperature=0):
    return OpenAI(temperature=temperature, openai_api_key=st.secrets["openai_key"] )

@st.cache_resource
def sf_engine():
    engine = create_engine(
        f"snowflake://{sf_user}:{sf_pw}@{sf_acct}/{sf_db}/{sf_schema}?warehouse={sf_wh}"
        )
    return engine

@st.cache_resource
def sf_connection():
    return engine.connect()

@st.cache_resource
def sql_db():
    return SQLDatabase(engine)

from langchain.prompts.prompt import PromptTemplate

_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct SNOWFLAKE query to run, then look at the results of the query and return the answer.
You will be answering questions about financial statements. So only consider the appropriate financial statement that a question would relate.
For instance, only use the income statement if a question is posed for revenue. 
Various tickers are all included in a single table for each of the three core financial statements and are located in these tables with the alias to be used next to each table in parentheses
INCOME_STATEMENT_ANNUAL (alias: isa), 
CASH_FLOW_STATEMENT_ANNUAL (alisas: cfs), 
BALANCE_SHEET_ANNUAL (alias: bsa)
Make sure any request is translated to one or more of these specific table names including the "_ANNUAL" suffix.
Column names do not have a space and they do not have an underscore so you will need to translate a request for'Net income' to something like netincome or free cash flow to freecashflow
In the event that a request is made that would come from multiple financial statements, then join based on the year and ticker columns. For instance, if the netincome and assets are necessary take netincome from the income statement and assets from the balance sheet statement and join based on year.
It is very important to create an alias for each table during the join and don't bring in year multiple times for the resulting output to avoid ambiguous names. Just choose one of the year columns instead
A user may request total assets in which this means totalassets column and not to sum assets

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Question: {input}"""
PROMPT = PromptTemplate(
    input_variables=["input"], template=_DEFAULT_TEMPLATE
)

@st.cache_data(ttl=600)
def db_chain(str_input):
    sf_chain = SQLDatabaseChain(llm=llm, database=sql_database, prompt=PROMPT, return_intermediate_steps=True)
    return sf_chain(str_input)

@st.cache_data(ttl=600)
def sf_query(str_input):
    return pd.read_sql(str_input, connection)

@st.cache_resource
def pinecone_init():
    return pinecone.init(
    api_key=st.secrets['pinecone_key'], 
    environment=st.secrets['pinecone_env'] 
    )

engine = sf_engine()
connection = sf_connection()
sql_database = sql_db()


tick_list = ['BRK.A','AAPL','PG','JNJ','MA','MCO','VZ','KO','AXP', 'BAC']
fin_statement_list = ['income_statement','balance_sheet','cash_flow_statement']

# create tabs
tab1, tab2, tab3 = st.tabs(["Data Exploration", "LLM-Powered Q/A ", "Model Building"])

with tab1:
    sel_tick = st.selectbox("Select a ticker to view", tick_list)
    sel_statement = st.selectbox("Select a ticker to view", fin_statement_list)
    df = sf_query(f"select * from financials.public.{sel_statement}_annual where ticker = '{sel_tick}'")
    st.dataframe(df)

with tab2:
    llm = llm_connection(temperature=0 )
    str_input = st.text_input(label='What would you like to answer? (e.g. What was the revenue and net income for Apple for the last 5 years?)')

    if len(str_input) > 1:
        output = db_chain(str_input)
        st.write(output['result'])
        st.write(sf_query(output['intermediate_steps'][0]))

        llm(f"Based on the output provided in {output['result']} produce python code to turn the data into a dataframe and plot the results but skip the import steps")
        #agent_executor.run(f"Based on the output provided in {output['result']}, create a dataframe of the contents and plot the results with an informative title")

    agent_executor = create_python_agent(
        llm=llm,
        tool=PythonREPLTool(),
        verbose=False
    )

    with tab3:
        pinecone_init()
        index_name = "buffett"
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_key"])

        docsearch = Pinecone.from_existing_index(index_name,embeddings)

        pdf_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0.1, model_name="gpt-3.5-turbo"),
                            docsearch, return_source_documents=True,  openai_api_key=st.secrets["openai_key"])

        query = st.text_input("What would you like to ask Warren? (e.g List reasons you purchased Apple and why it has worked out so well?)")
        if len(query)>1:
            result = pdf_qa({"question": query, "chat_history": ""})
            #print(result)
            st.write(result['answer'])
            st.markdown("Additional details from the search result")
            st.write(result)


