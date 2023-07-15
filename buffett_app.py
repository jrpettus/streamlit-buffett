import snowflake.connector
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import prompts
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
#from langchain.chains import ChatVectorDBChain # for chatting with the pdf
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone # for the vectorization part
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import (
    RetrievalQA
)

st.set_page_config(layout="wide")

# grab some of the secrets
sf_db = st.secrets["sf_database"]
sf_schema = st.secrets["sf_schema"]

# establish snowpark connection
conn = st.experimental_connection("snowpark")

# Reset the connection before using it if it isn't healthy
# Note: is_healthy() isn't a real method and is just shown for example here.
try:
    query_test = conn.query('select net_income from {database}.{schema}.income_statement_annual limit 1')
except:
    conn.reset()

@st.cache_data()
def pull_financials(database, schema, statement, ticker):
    """
    query to pull financial data from snowflake based on database, schema, statemen and ticker
    """
    df = conn.query(f"select * from {database}.{schema}.{statement} where ticker = '{ticker}' order by year desc")
    df.columns = [col.lower() for col in df.columns]
    return df

# metrics for kpi cards
def kpi_recent(df, metric, periods=2, unit=1000000000):
    """
    filters a financial statement dataframe down to the most recent periods
    df is the financial statement. Metric is the column to be used.
    """
    return df.sort_values('year',ascending=False).head(periods)[metric]/unit

tick_list = ['BRK.A','AAPL','PG','JNJ','MA','MCO','VZ','KO','AXP', 'BAC']
fin_statement_list = ['income_statement','balance_sheet','cash_flow_statement']

# create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Financial Statement Natural Language Querying :dollar:", "Financial Data Exploration :chart_with_upwards_trend:",
                                  "Shareholder Letter Natural Language Querying :memo:", "Additional Details :notebook:"])

with st.sidebar:
    st.markdown("""
    # Ask the Oracle of Omaha: Using LLMs to Provide a View into the World of Warren Buffett :moneybag:
    This app enables exploration into the World of Warren Buffett, enabling you to ask financial questions regarding his top investments and over 40 years of his shareholder letters.
    This app is powered by Snowflake :snowflake:, Streamlit, OpenAI, Langchain and Pinecone, leveraging Large Language Models (LLMs)

    Tabs:
    ### 1: Financial Statement Natural Language Querying :dollar:
    Ask financial questions using natural language regarding the investments
    ### 2: Financial Data Exploration :chart_with_upwards_trend:
    Query Snowflake to view financials for various Warren Buffett investments
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
    # Natural Language Financials Querying :dollar:
    ### Leverage LLMs to translate natural language questions related to financial statements and turn those into direct Snowflake queries
    Data is stored and queried directly from income statement, balance sheet, and cash flow statement in Snowflake

    **Example questions to ask:**

    - What was the net income from 1996 through 2000 for Proctor and Gamble?
    - What was the revenue, depreciation, and total liabilities for Apple for the last 5 years?
    - Rank the companies in descending order based on their net income in 2022. Include the ticker and net income value
    - What has been the average for total assets and total liabilities for each company over the last 3 years? List the top 3
    """
    )
    
    #chain = prompts.load_chain()
    str_input = st.text_input(label='What would you like to answer? (e.g. What was the revenue and net income for Apple for the last 5 years?)')

    if len(str_input) > 1:
        with st.spinner('Looking up your question in Snowflake now...'):
            prompts.execute_chain(str_input)
            try:
                output = prompts.execute_chain(str_input)
                try:
                    # if the output doesn' work we will try one additional attempt to fix it
                    st.write(output)
                    st.write(conn.query(output['result']))
                except:
                    st.write("The first attempt didn't pull what you were needing. Trying again...")
                    output = prompts.execute_chain(f'You need to fix the code. If the question is complex, consider using one or more CTE. Also, examine the DDL statements and try to correct this question/query: {output}')
                    st.write(output)
                    st.write(conn.query(output['result']))
            except:
                st.write("Please try to improve your prompt or provide feedback on the error encountered")

    with tab2:
        st.markdown("""
        # Financial Data Exploration :chart_with_upwards_trend:
    
        View financial statement data for selected companies owned by Warren Buffet by querying Snowflake directly.
    
        Available companies are identified in the selection drop down
        """)
        sel_tick = st.selectbox("Select a ticker to view", tick_list)
    
        inc_st = pull_financials(sf_db, sf_schema, 'income_statement_annual', sel_tick)
        bal_st = pull_financials(sf_db, sf_schema, 'balance_sheet_annual', sel_tick)
        bal_st['debt_to_equity'] = bal_st['total_debt'].div(bal_st['total_equity'])
        cf_st =  pull_financials(sf_db, sf_schema, 'cash_flow_statement_annual', sel_tick) 
        
        # find the most recent 2 periods
        net_inc = kpi_recent(inc_st, 'net_income')
        net_inc_ratio = kpi_recent(inc_st, 'net_income_ratio', periods=2, unit=1)
        fcf = kpi_recent(cf_st, 'free_cash_flow' )
        debt_ratio = kpi_recent(bal_st, 'debt_to_equity', periods=2, unit=1)
      
        col1, col2 = st.columns((1,1))
        # year cutoff
        year_cutoff = 20
    
        with col1:
            # Net Income metric
            st.metric('Net Income', f'${net_inc[0]}B', delta=round(net_inc[0]-net_inc[1],2), delta_color="normal", help=None, label_visibility="visible")
            st.altair_chart(alt.Chart(inc_st.head(year_cutoff)).mark_bar().encode(
                x='year',
                y='net_income'
                ).properties(title="Net Income")
            ) 
            
            # netincome ratio
            st.metric('Net Profit Margin', f'{round(net_inc_ratio[0]*100,2)}%', delta=round(net_inc_ratio[0]-net_inc_ratio[1],2), delta_color="normal", help=None, label_visibility="visible")
            st.altair_chart(alt.Chart(inc_st.head(year_cutoff)).mark_bar().encode(
                x='year',
                y='net_income_ratio'
                ).properties(title="Net Profit Margin")
            ) 
        
        with col2:
            # free cashflow
            st.metric('Free Cashflow', f'${fcf[0]}B', delta=round(fcf[0]-fcf[1],2), delta_color="normal", help=None, label_visibility="visible")
            st.altair_chart(alt.Chart(cf_st.head(year_cutoff)).mark_bar().encode(
                x='year',
                y='free_cash_flow'
                ).properties(title="Free Cash Flow")
            ) 

            # debt to equity
            st.metric('Debt to Equity', f'{round(debt_ratio[0],2)}', delta=round(debt_ratio[0]-debt_ratio[1],2), delta_color="normal", help=None, label_visibility="visible")
            st.altair_chart(alt.Chart(bal_st.head(year_cutoff)).mark_bar().encode(
                x='year',
                y='debt_to_equity'
                ).properties(title="Debt to Equity")
            ) 
    
        sel_statement = st.selectbox("Select a statement to view", fin_statement_list)
        fin_statement_dict = {'income_statement': inc_st, 'balance_sheet': bal_st, 'cash_flow_statement':cf_st}
        st.dataframe(fin_statement_dict[sel_statement])
    
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

        query = st.text_input("What would you like to ask Warren Buffett?")
        if len(query)>1:
            with st.spinner('Looking through lots of Shareholder letters now...'):
                try:
                    result = prompts.pdf_question(query)
                    st.write(result['answer'])
                    st.write("Source Document Detail")
                    st.write(result['source_documents'][0])
                except:
                    st.write("Please try to improve your prompt or provide feedback on the error encountered")

with tab4:
    st.markdown("""
    
    Additional Details:

    - Tabs 1 and 3 can likely be consolidated leveraging Langchain "tools" with better prompting templates.

    """)
