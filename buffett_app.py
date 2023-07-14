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
#@st.cache_resource
def llm_connection(temperature=0):
    return OpenAI(temperature=temperature, openai_api_key=st.secrets["openai_key"] )

#@st.cache_resource
def sf_engine():
    engine = create_engine(
        f"snowflake://{sf_user}:{sf_pw}@{sf_acct}/{sf_db}/{sf_schema}?warehouse={sf_wh}"
        )
    return engine

#@st.cache_resource
def sf_connection():
    return engine.connect()

#@st.cache_resource
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
Only use the alias after using the full table name and only include the relevant financial statements.
Make sure any request is translated to one or more of these specific table names including the "_ANNUAL" suffix.
In the event that a request is made that would come from multiple financial statements, then join based on the year and ticker columns. For instance, if the net_income and assets are necessary take netincome from the income statement and assets from the balance sheet statement and join based on year.
It is very important to create an alias for each table during the join and don't bring in year multiple times for the resulting output to avoid ambiguous names. Just choose one of the year columns instead
A user may request total assets in which this means total_assets column and not to sum assets

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Question: {input}"""
PROMPT = PromptTemplate(
    input_variables=["input"], template=_DEFAULT_TEMPLATE
)

#@st.cache_data(ttl=600)
def db_chain(str_input):
    sf_chain = SQLDatabaseChain(llm=llm, database=sql_database, prompt=PROMPT, return_intermediate_steps=True)
    return sf_chain(str_input)

#@st.cache_data(ttl=600)
def sf_query(str_input):
    return pd.read_sql(str_input, connection)

#@st.cache_resource
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
tab1, tab2, tab3, tab4 = st.tabs(["Financial Data Exploration :chart_with_upwards_trend:", "Financial Statement Natural Language Querying :dollar:", 
                                  "Shareholder Letter Natural Language Querying :memo:", "Additional Details :notebook:"])

with st.sidebar:
    st.markdown("""
    # Ask the Oracle of Omaha: Using LLMs to Provide a View into the World of Warren Buffett :moneybag:
    This app enables exploration into the World of Warren Buffett, enabling you to ask financial questions regarding his top investments and over 40 years of his shareholder letters.
    This app is powered by Snowflake :snowflake:, Streamlit, OpenAI, Langchain and Pinecone, leveraging Large Language Models (LLMs)

    Tabs:

    ### 1: Financial Data Exploration :chart_with_upwards_trend:
    Query Snowflake to view financials for various Warren Buffett investments
    ### 2: Financial Statement Natural Language Querying :dollar:
    Ask financial questions using natural language regarding the investments
    ### 3: Shareholder Letter Natural Language Querying :memo:
    Ask various questions based on Warren Buffett's shareholder letters from 1977 through 2022  

    **Current Available Companies to ask financials about for tabs 1 and 2:**
    1. Apple
    2. Bershire Hathaway
    3. Proctor and Gamble
    4. Johnson and Johnson
    5. Mastercard
    6. Moodys Corp
    7. Verizon
    8. American Express
    9. Bank of America  
    """)

with tab1:
    st.markdown("""
    # Financial Data Exploration :chart_with_upwards_trend:

    View financial statement data for selected companies owned by Warren Buffet by querying Snowflake directly.

    Available companies are identified in the selection drop down
    """)
    sel_tick = st.selectbox("Select a ticker to view", tick_list)

    inc_st = sf_query(f"select * from {sf_db}.{sf_schema}.income_statement_annual where ticker = '{sel_tick}' order by year desc")
    bal_st = sf_query(f"select * from {sf_db}.{sf_schema}.balance_sheet_annual where ticker = '{sel_tick}' order by year desc")
    bal_st['debt_to_equity'] = bal_st['total_debt'].div(bal_st['total_equity'])
    cf_st = sf_query(f"select * from {sf_db}.{sf_schema}.cash_flow_statement_annual where ticker = '{sel_tick}' order by year desc")

    # metrics for kpi cards
    def kpi_recent(df, metric, periods=2, unit=1000000000):
        return df.sort_values('year',ascending=False).head(periods)[metric]/unit
    
    # find the most recent 2 periods
    net_inc = kpi_recent(inc_st, 'net_income')
    net_inc_ratio = kpi_recent(inc_st, 'net_income_ratio', periods=2, unit=1)
    fcf = kpi_recent(cf_st, 'free_cash_flow' )
    debt_ratio = kpi_recent(bal_st, 'debt_to_equity', periods=2, unit=1)
  
    col1, col2 = st.columns((1,1))
    # year cutoff
    year_cutoff = 20

    with col1:
        #st.subheader("Net Income")
        st.metric('Net Income', f'${net_inc[0]}B', delta=round(net_inc[0]-net_inc[1],2), delta_color="normal", help=None, label_visibility="visible")
        st.altair_chart(alt.Chart(inc_st.head(year_cutoff)).mark_bar().encode(
            x='year',
            y='net_income'
            ).properties(title="Net Income")
        ) 
        
        #st.subheader("Net Profit Margin")
        # netincome ratio
        st.metric('Net Profit Margin', f'{round(net_inc_ratio[0]*100,2)}%', delta=round(net_inc_ratio[0]-net_inc_ratio[1],2), delta_color="normal", help=None, label_visibility="visible")
        st.altair_chart(alt.Chart(inc_st.head(year_cutoff)).mark_bar().encode(
            x='year',
            y='net_income_ratio'
            ).properties(title="Net Profit Margin")
        ) 
    
    with col2:
        #st.subheader("Free Cashflow")
        # free cashflow
        st.metric('Free Cashflow', f'${fcf[0]}B', delta=round(fcf[0]-fcf[1],2), delta_color="normal", help=None, label_visibility="visible")
        st.altair_chart(alt.Chart(cf_st.head(year_cutoff)).mark_bar().encode(
            x='year',
            y='free_cash_flow'
            ).properties(title="Free Cash Flow")
        ) 

        st.metric('Debt to Equity', f'{round(debt_ratio[0],2)}', delta=round(debt_ratio[0]-debt_ratio[1],2), delta_color="normal", help=None, label_visibility="visible")
        st.altair_chart(alt.Chart(bal_st.head(year_cutoff)).mark_bar().encode(
            x='year',
            y='debt_to_equity'
            ).properties(title="Debt to Equity")
        ) 

    sel_statement = st.selectbox("Select a statement to view", fin_statement_list)
    fin_statement_dict = {'income_statement': inc_st, 'balance_sheet': bal_st, 'cash_flow_statement':cf_st}
    st.dataframe(fin_statement_dict[sel_statement])

with tab2:
    st.markdown("""
    # Natural Language Financials Querying :dollar:
    ### Leverage LLMs to translate natural language questions related to financial statements and turn those into direct Snowflake queries
    Data is stored and queried directly from income statement, balance sheet, and cash flow statement in Snowflake

    **Example questions to ask:**

    - What was the net income in 1996 through 2000 for Proctor and Gamble?
    - What was the revenue vs. total assets for Apple for the last 5 years?
    """
    )

    llm = llm_connection(temperature=0, )
    str_input = st.text_input(label='What would you like to answer? (e.g. What was the revenue and net income for Apple for the last 5 years?)')

    if len(str_input) > 1:
        with st.spinner('Looking up your question in Snowflake now...'):
            try:
                output = db_chain(str_input)
                st.write(output['result'])
                st.dataframe(sf_query(output['intermediate_steps'][1]))
                #st.write(output)
            except:
                st.write("Please try to improve your prompt or provide feedback on the error encountered")

        #llm(f"Based on the output provided in {output['result']} produce python code to turn the data into a dataframe and plot the results but skip the import steps")
        #agent_executor.run(f"Based on the output provided in {output['result']}, create a dataframe of the contents and plot the results with an informative title")

    #agent_executor = create_python_agent(
    #    llm=llm,
    #    tool=PythonREPLTool(),
    #    verbose=False
    #)

    with tab3:
        st.markdown("""
        # Shareholder Letter Natural Language Querying :memo:
        ### Ask questions from all of Warren Buffett's annual shareholder letters dating back to 1977

        These letters are much anticipated by investors for the wealth of knowledge that Buffett provides.
        The tool allows you to interact with these letters by asking questions and a LLM is used to find relevant answers.

        **Example questions to ask:**

        - Why has Apple been a good investment? What specific reasons has it increased in value?
        - What are some of your biggest lessons learned?
        - What do you look for in managers?
        - Are markets efficient? Give a specific example that you have used in a letter.
        """
        )

        pinecone_init()
        index_name = "buffett"
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_key"])

        docsearch = Pinecone.from_existing_index(index_name,embeddings)

        pdf_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0.1, model_name="gpt-3.5-turbo",openai_api_key=st.secrets["openai_key"]),
                            docsearch, return_source_documents=True)
        
        @st.cache_data(ttl=600)
        def pdf_question(query):
            return pdf_qa({"question": query, "chat_history": ""})

        query = st.text_input("What would you like to ask Warren Buffett?")
        if len(query)>1:
            with st.spinner('Looking through lots of Shareholder letters now...'):
                try:
                    result = pdf_question(query)
                    #print(result)
                    st.write(result['answer'])
                    #st.markdown("Additional details from the search result")
                    #st.write(result)
                except:
                    st.write("Please try to improve your prompt or provide feedback on the error encountered")

with tab4:
    st.markdown("""
    
    Additional Details:

    - Tabs 2 and 3 can likely be consolidated leveraging Langchain "tools" with better prompting templates.

    """)
