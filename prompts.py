import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pinecone


FS_TEMPLATE = """ You are an expert SQL developer querying about financials statements. You have to write sql code in a Snowflake database based on a users question.
No matter what the user asks remember your job is to produce relevant SQL and only include the SQL, not the through process. So if a user asks to display something, you still should just produce SQL.
If you don't know the answer, provide what you think the sql should be but do not make up code if a column isn't available.

As an example, a user will ask "Display the last 5 years of net income for Johnson and Johnson?" The SQL to generate this would be:

select year, net_income
from financials.prod.income_statement_annual
where ticker = 'JNJ'
order by year desc
limit 5;

Questions about income statement fields should query financials.prod.income_statement_annual
Questions about balance sheet fields (assets, liabilities, etc.) should query  financials.prod.balance_sheet_annual
Questions about cash flow fields (operating cash, investing activities, etc.) should query financials.prod.cash_flow_statement_annual

The financial figure column names include underscores _, so if a user asks for free cash flow, make sure this is converted to FREE_CASH_FLOW. 
Some figures may have slightly different terminology, so find the best match to the question. For instance, if the user asks about Sales and General expenses, look for something like SELLING_AND_GENERAL_AND_ADMINISTRATIVE_EXPENSES

If the user asks about multiple figures from different financial statements, create join logic that uses the ticker and year columns. Don't use SQL terms for the table alias though. Just use a, b, c, etc.
The user may use a company name so convert that to a ticker.

Question: {question}
Context: {context}

SQL: ```sql ``` \n
 
"""
FS_PROMPT = PromptTemplate(input_variables=["question", "context"], template=FS_TEMPLATE, )

LETTER_TEMPLATE = """ You are tasked with retreiving questions regarding Warren Buffett from his shareholder letters.
Provide an answer based on this retreival, and if you can't find anything relevant, just say "I'm sorry, I couldn't find that."
{context}
Question: {question}
Anwer:
 
"""
LETTER_PROMPT = PromptTemplate(input_variables=["question", "context"], template=LETTER_TEMPLATE, )

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    max_tokens=1000, 
    openai_api_key=st.secrets["openai_key"]
)


def get_faiss():
    " get the loaded FAISS embeddings"
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_key"])
    return FAISS.load_local("faiss_index", embeddings)


def get_pinecone():
    " get the pinecone embeddings"
    pinecone.init(
        api_key=st.secrets['pinecone_key'], 
        environment=st.secrets['pinecone_env'] 
        )
    
    index_name = "buffett"
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_key"])
    return Pinecone.from_existing_index(index_name,embeddings)


def fs_chain(question):
    """
    returns a question answer chain for faiss vectordb
    """

    docsearch = get_faiss()
    qa_chain = RetrievalQA.from_chain_type(llm, 
                                           retriever=docsearch.as_retriever(),
                                           chain_type_kwargs={"prompt": FS_PROMPT})
    return qa_chain({"query": question})


def letter_chain(question):
    """returns a question answer chain for pinecone vectordb"""
    
    docsearch = get_pinecone()
    retreiver = docsearch.as_retriever(#
        #search_type="similarity", #"similarity", "mmr"
        search_kwargs={"k":3}
    )
    qa_chain = RetrievalQA.from_chain_type(llm, 
                                            retriever=retreiver,
                                           chain_type="stuff", #"stuff", "map_reduce","refine", "map_rerank"
                                           return_source_documents=True,
                                           #chain_type_kwargs={"prompt": LETTER_PROMPT}
                                          )
    return qa_chain({"query": question})


def letter_qa(query, temperature=.1,model_name="gpt-3.5-turbo"):
    """
    this method was deprecated but seems to be more efficient from a token perspective
    """
    pdf_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=temperature, model_name=model_name, openai_api_key=st.secrets["openai_key"]),
                    pinecone_search(), return_source_documents=True)
    return pdf_qa({"question": query, "chat_history": ""})

