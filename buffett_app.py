import snowflake.connector
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import prompts

st.set_page_config(layout="wide")

# Variables
sf_db = st.secrets["sf_database"]
sf_schema = st.secrets["sf_schema"]
tick_list = ['BRK.A','AAPL','PG','JNJ','MA','MCO','VZ','KO','AXP', 'BAC']
fin_statement_list = ['income_statement','balance_sheet','cash_flow_statement']
year_cutoff = 20 # year cutoff for financial statement plotting

# establish snowpark connection
conn = st.connection("snowflake")

# Reset the connection before using it if it isn't healthy
try:
    query_test = conn.query('select 1')
except:
    conn.reset()

@st.cache_data
def pull_financials(database, schema, statement, ticker):
    """
    query to pull financial data from snowflake based on database, schema, statement and ticker
    """
    df = conn.query(f"select * from {database}.{schema}.{statement} where ticker = '{ticker}' order by year desc")
    df.columns = [col.lower() for col in df.columns]
    return df

# metrics for kpi cards
@st.cache_data
def kpi_recent(df, metric, periods=2, unit=1000000000):
    """
    filters a financial statement dataframe down to the most recent periods
    df is the financial statement. Metric is the column to be used.
    """
    return df.sort_values('year',ascending=False).head(periods)[metric]/unit

def plot_financials(df, x, y, x_cutoff, title):
    """"
    helper to plot the altair financial charts
    """
    return st.altair_chart(alt.Chart(df.head(x_cutoff)).mark_bar().encode(
        x=x,
        y=y
        ).properties(title=title)
    ) 

# adding this to test out caching
st.cache_data(ttl=86400)
def fs_chain(str_input):
    """
    performs qa capability for a question using sql vector db store
    the prompts.fs_chain is used but with caching
    """
    output = prompts.fs_chain(str_input)
    return output

# adding this to test out caching
st.cache_data(ttl=86400)
def sf_query(str_input):
    """
    performs snowflake query with caching
    """
    data = conn.query(str_input)
    return data

# create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Financial Statement Questions :dollar:", 
    "Financial Data Exploration :chart_with_upwards_trend:",
    "Shareholder Letter Questions :memo:", 
    "Additional Details :notebook:"]
    )

with st.sidebar:
    st.markdown("""
    # Ask the Oracle of Omaha: Using LLMs to Provide a View into the World of Warren Buffett :moneybag:
    This app enables exploration into the World of Warren Buffett, enabling you to ask financial questions regarding his top investments and over 40 years of his shareholder letters.
    This app is powered by Snowflake :snowflake:, Streamlit, OpenAI, Langchain and Pinecone, leveraging Large Language Models (LLMs)

    Tabs:
    ### 1: Financial Statement Questions :dollar:
    Ask financial questions using natural language regarding the investments
    ### 2: Financial Data Exploration :chart_with_upwards_trend:
    Query Snowflake to view financials for various Warren Buffett investments
    ### 3: Shareholder Letter Questions :memo:
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

    Produced by @randypettus
    """)

with tab1:
    st.markdown("""
    # Financial Statement Questions :dollar:
    ### Leverage LLMs to translate natural language questions related to financial statements and turn those into direct Snowflake queries
    Data is stored and queried directly from income statement, balance sheet, and cash flow statement in Snowflake

    **Example questions to ask:**

    - What was Proctor and Gamble's net income from 2010 through 2020?
    - What was Apple's debt to equity ratio for the last 5 years?
    - Rank the companies in descending order based on their net income in 2022. Include the ticker and net income value
    - What has been the average for total assets and total liabilities for each company over the last 3 years? List the top 3
    """
    )
    
    str_input = st.text_input(label='What would you like to answer? (e.g. What was the revenue and net income for Apple for the last 5 years?)')

    if len(str_input) > 1:
        with st.spinner('Looking up your question in Snowflake now...'):
            try:
                output = fs_chain(str_input)
                #st.write(output)
                try:
                    # if the output doesn't work we will try one additional attempt to fix it
                    query_result = sf_query(output['result'])
                    if len(query_result) > 1:
                        st.write(query_result)
                        st.write(output)
                except:
                    st.write("The first attempt didn't pull what you were needing. Trying again...")
                    output = fs_chain(f'You need to fix the code but ONLY produce SQL code output. If the question is complex, consider using one or more CTE. Examine the DDL statements and answer this question: {output}')
                    st.write(sf_query(output['result']))
                    st.write(output)
            except:
                st.write("Please try to improve your question. Note this tab is for financial statement questions. Use Tab 3 to ask from shareholder letters. Also, only a handful of companies are available, which you can see on the side bar.")
                st.write(f"Final errored query used:")
                st.write(output)

with tab2: 
    st.markdown("""
    # Financial Data Exploration :chart_with_upwards_trend:

    View financial statement data for selected companies owned by Warren Buffet by querying Snowflake directly.

    Available companies are identified in the selection drop down
    """)
    sel_tick = st.selectbox("Select a ticker to view", tick_list)

    # pull the financial statements
    # This whole section could be more efficient...
    inc_st = pull_financials(sf_db, sf_schema, 'income_statement_annual', sel_tick)
    bal_st = pull_financials(sf_db, sf_schema, 'balance_sheet_annual', sel_tick)
    bal_st['debt_to_equity'] = bal_st['total_debt'].div(bal_st['total_equity'])
    cf_st =  pull_financials(sf_db, sf_schema, 'cash_flow_statement_annual', sel_tick) 
  
    col1, col2 = st.columns((1,1))
    with col1:
        # Net Income metric
        net_inc = kpi_recent(inc_st, 'net_income')
        st.metric('Net Income', 
                  f'${net_inc[0]}B', 
                  delta=round(net_inc[0]-net_inc[1],2),
                  delta_color="normal", 
                  help=None, 
                  label_visibility="visible")
        plot_financials(inc_st, 'year', 'net_income', year_cutoff, 'Net Income')
        
        # netincome ratio
        net_inc_ratio = kpi_recent(inc_st, 'net_income_ratio', periods=2, unit=1)
        st.metric('Net Profit Margin', 
                  f'{round(net_inc_ratio[0]*100,2)}%',
                  delta=round(net_inc_ratio[0]-net_inc_ratio[1],2), 
                  delta_color="normal", 
                  help=None, 
                  label_visibility="visible")
        plot_financials(inc_st, 'year', 'net_income_ratio', year_cutoff, 'Net Profit Margin')
    
    with col2:
        # free cashflow
        fcf = kpi_recent(cf_st, 'free_cash_flow' )
        st.metric('Free Cashflow', 
                  f'${fcf[0]}B', 
                  delta=round(fcf[0]-fcf[1],2), 
                  delta_color="normal", 
                  help=None, 
                  label_visibility="visible")
        plot_financials(cf_st, 'year', 'free_cash_flow', year_cutoff, 'Free Cash Flow')

        # debt to equity
        debt_ratio = kpi_recent(bal_st, 'debt_to_equity', periods=2, unit=1)
        st.metric('Debt to Equity', 
                  f'{round(debt_ratio[0],2)}', 
                  delta=round(debt_ratio[0]-debt_ratio[1],2), 
                  delta_color="normal", 
                  help=None, 
                  label_visibility="visible")
        plot_financials(bal_st, 'year', 'debt_to_equity', year_cutoff, 'Debt to Equity')

    # enable a financial statment to be selected and viewed
    sel_statement = st.selectbox("Select a statement to view", fin_statement_list)
    fin_statement_dict = {'income_statement': inc_st,
                          'balance_sheet': bal_st, 
                          'cash_flow_statement':cf_st}
    st.dataframe(fin_statement_dict[sel_statement])

with tab3:
    st.markdown("""
    # Shareholder Letter Questions :memo:
    ### Ask questions from all of Warren Buffett's annual shareholder letters dating back to 1977

    These letters are much anticipated by investors for the wealth of knowledge that Buffett provides.
    The tool allows you to interact with these letters by asking questions and a LLM is used to find relevant answers.

    **Example questions to ask:**
    
    - How does your view on managing investment risk differ from professional investment managers?
    - What was your gain in net worth in 1984?
    - What are some of your biggest lessons learned?
    - What do you look for in managers?
    - Are markets efficient? Give a specific example that you have used in a letter.
    """
    )

    query = st.text_input("What would you like to ask Warren Buffett?")
    if len(query)>1:
        with st.spinner('Looking through lots of Shareholder letters now...'):
            
            try:
                st.caption(":blue[Warren's response] :sunglasses:")
                #st.write(prompts.letter_qa(query))
                result = prompts.letter_chain(query)
                st.write(result['result'])
                st.caption(":blue[Source Documents Used] :📄:")
                st.write(result['source_documents'])
            except:
                st.write("Please try to improve your question")

with tab4:
    st.markdown("""
    Additional Details:
    - Tabs 1 and 3 can likely be consolidated leveraging Langchain "tools" with better prompting templates.
    """)
    st.subheader("App Architecture")
    
    from PIL import Image
    image = Image.open('./assets/buffett-app-architecture.png')
    st.image(image, caption='Buffett app architecture')
