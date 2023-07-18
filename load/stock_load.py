import os
import glob
import numpy as np
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from snowflake.snowpark.session import Session
import streamlit as st

# snowpark connection
CONNECTION_PARAMETERS = {
   "account": st.secrets['account'], 
   "user": st.secrets['user'],
   "password": st.secrets['password'],
    "database": st.secrets['database'],
   "schema": st.secrets['schema'],
   "warehouse": st.secrets['warehouse'], 
}

# create session
session = Session.builder.configs(CONNECTION_PARAMETERS).create()

# create a list of the statements which should match the folder name
statements = ['INCOME_STATEMENT_ANNUAL','BALANCE_SHEET_ANNUAL','CASH_FLOW_STATEMENT_ANNUAL']

# Load data into snowflake by looping through the csv files
for statement in statements:
    path = f'./load/financials/{statement.lower()}/' 
    files = glob.glob(os.path.join(path, "*.csv"))
    df = pd.concat((pd.read_csv(f) for f in files))
    print(statement)
    # note that overwrite is used to start. If adding future data, move to append with upsert process
    session.create_dataframe(df).write.mode('overwrite').save_as_table(statement)

# automatically get the ddl from the created tables
# create empty string that will be populated
ddl_string = ''

# run through the statements and get ddl
for statement in statements:
    ddl_string += session.sql(f"select get_ddl('table', '{statement}')").collect()[0][0] + '\n\n'
    
ddl_file = open("ddls.sql", "w")
n = ddl_file.write(ddl_string)
ddl_file.close()
