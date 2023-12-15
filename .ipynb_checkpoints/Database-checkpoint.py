from __future__ import annotations

# Imports
import re
import pandas as pd
import numpy as np
import uuid
import datetime
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import os
import glob
import sqlite3

np.random.seed(42)

"""
Retrieve the most recent transactions and transaction details CSV files from the specified directory.

Args:
- directory (str): Directory path where transactions CSV files are stored (default: 'Data\\Simulations').

Returns:
- transactions_df (DataFrame): DataFrame containing the most recent transactions if the simulation has been run.        
- transactions_details_df (DataFrame): DataFrame containing the most recent transaction details if the simulation has been run.
"""
def get_most_recent_sim(directory='Data\\Simulations'):

    
    most_recent_files = {'transactions': None, 'transactions_details': None}

    # Find files matching the pattern for transactions
    transactions_files = glob.glob(os.path.join(directory, 'transactions_s*.csv'))
    if transactions_files:
        most_recent_files['transactions'] = max(transactions_files, key=os.path.getctime)

    # Find files matching the pattern for transactions_details
    transactions_details_files = glob.glob(os.path.join(directory, 'transactions_details_s*.csv'))
    if transactions_details_files:
        most_recent_files['transactions_details'] = max(transactions_details_files, key=os.path.getctime)
    #print(most_recent_files)
    transactions_df = pd.read_csv(most_recent_files['transactions'])
    transactions_details_df = pd.read_csv(most_recent_files['transactions_details'])
    return transactions_df,transactions_details_df


def delete_existing_db(db_file='store.db'):
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"Deleted the existing database file: {db_file}")
    else:
        print(f"The database file {db_file} does not exist.")

def make_db(directory='Data\\Simulations', 
            transaction_file = None, 
            transaction_details_file = None):
    
    """
    Create an SQLite database from transaction and customer/product data, and store the data in the respective tables.

    Args:
    - directory (str): Directory path where transactions and customer/product CSV files are stored (default: 'Data\\Simulations').
    - transaction_file (str): File path of the transaction CSV file (default: None).
    - transaction_details_file (str): File path of the transaction details CSV file (default: None).

    Returns:
    None
    """

    if transaction_file is None and transaction_details_file is None:
        transactions_df, transactions_details_df = get_most_recent_sim(directory)
    else:
        transactions_df = pd.read_csv(transaction_file)
        transactions_details_df = pd.read_csv(transaction_details_file)
    customers_file = 'Data/customers.csv'
    products_file = 'Data/products.csv'
    
    customers_df = pd.read_csv(customers_file)
    products_df = pd.read_csv(products_file)
    
    delete_existing_db()
    conn = sqlite3.connect('store.db')  
    cursor = conn.cursor()
    create_transactions_query = '''
        CREATE TABLE transactions(
            transaction_id TEXT PRIMARY KEY,
            customer_id TEXT,
            purchase_time TEXT,
            total_price REAL
        )
        '''
    create_transactions_details_query = '''
        CREATE TABLE transactions_details(
            transaction_id TEXT PRIMARY KEY,
            product_id INT,
            product_price REAL,
            product_sale BOOLEAN,
            product_display BOOLEAN)     
    
        '''
    create_products_query = '''
        CREATE TABLE products(
            product_id INT PRIMARY KEY,
            category TEXT,
            price REAL,
            on_sale BOOLEAN,
            on_display BOOLEAN
        )
    
    '''
    
    create_customers_query = '''
        CREATE TABLE customers(
            customer_id INT PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            age INT,
            gender TEXT,
            location INT,
            buyer_habit TEXT,
            buyer_spending TEXT
            )
            '''
    cursor.execute(create_products_query)
    cursor.execute(create_customers_query)
    cursor.execute(create_transactions_query)
    cursor.execute(create_transactions_details_query)
    conn.commit()
    
    try:
        products_df.to_sql(name = 'products', con=conn, if_exists='replace', index=False)
        customers_df.to_sql(name = 'customers', con=conn, if_exists='replace', index=False)
        transactions_df.to_sql(name ='transactions', con=conn, if_exists='replace', index=False)
        transactions_details_df.to_sql(name='transactions_details', con=conn, if_exists='replace', index=False)
        conn.commit()
        
    except Exception as e:
        print(f"Error inserting data into tables: {e}")
  
    finally:
        conn.close()
        print('Database was created and now the connection is closed.')