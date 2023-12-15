from __future__ import annotations

import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import os
import glob
import sqlite3

def save_features_to_csv(data, filename_prefix='cust_features_', directory="Data/Simulations/Customer_Features"):
   # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Construct the file path
    file_path = os.path.join(directory, f"{filename_prefix}{time.time()}.csv")

    # Save the data to CSV
    data.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

"""
Feature Engineering class for analyzing customer data and performing various analytics on it.

Args:
- db_file (str): Path to the SQLite database file (default: 'store.db').

Methods:
- __init__: Initializes the FeatEng class and establishes a connection to the database.
- merge_features: Merges different features and creates a comprehensive DataFrame.
- total_sales_by_segment: Visualizes the total spending by buyer habit and spending segment.
- scatter_with_spending: Creates a scatterplot to analyze the correlation between a specific feature and total spending.
- close_connection: Closes the connection to the SQLite database.
- get_customer_data: Retrieves customer data from the database.
- purchase_frequency: Calculates the purchase frequency for each customer.
- total_spent_by_customers: Calculates the total amount spent by each customer.
- avg_cost_by_customers: Calculates the average cost spent by each customer.
- avg_var_by_customers: Calculates the average variance in spending by each customer.
- category_distribution_by_customers: Analyzes the distribution of purchased product categories by customers.
- payment_distribution_by_customers: Analyzes the distribution of payment types used by customers.
- avg_price_per_product: Calculates the average price per product purchased by customers.
- product_distribution_by_customers: Analyzes the distribution of product sale and display attributes by customers.
"""
class FeatEng:

    def __init__(self, db_file='store.db'):
        # Establish a connection to the SQL database
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.features = pd.DataFrame() # make empty dataframe
        self.features_df = self.merge_features()
        self.sales_by_segment = self.total_sales_by_segment()
        self.age_spending_corr = self.scatter_with_spending('age')
        self.close_connection()
        
    def close_connection(self):
        self.conn.close() 
    
    def get_customer_data(self):
        # get the custoemr data
        query = """
            SELECT *
            FROM customers
        """
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        
        # convert sql query result to a DataFrame
        columns = ['customer_id','first_name',
                   'last_name','age','gender',
                   'buyer_habit',
                   'buyer_spending', 'zip','state_name']
        customer_df = pd.DataFrame(result, columns=columns)
        return customer_df
    
    def purchase_frequency(self):
        query = '''
            SELECT customer_id, COUNT(*) AS purchase_count
            FROM transactions
            GROUP BY customer_id
        '''
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        columns = ['customer_id', 'purchase_count']
        purchase_freq_df = pd.DataFrame(result, columns = columns)
        return purchase_freq_df
    
    def total_spent_by_customers(self):
        query = '''
            SELECT customer_id, SUM(total_price) AS total_spent
            FROM transactions
            GROUP BY customer_id
        '''
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        columns = ['customer_id', 'total_spent']
        totals_df = pd.DataFrame(result,columns = columns)
        return totals_df
    
    def avg_cost_by_customers(self):
        query = '''
            SELECT customer_id, AVG(total_price) AS avg_cost
            FROM transactions
            GROUP BY customer_id
        '''
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        columns = ['customer_id', 'avg_spent']
        avg_df = pd.DataFrame(result,columns = columns)
        return avg_df

    def avg_var_by_customers(self):
        query = '''
            SELECT customer_id, AVG((total_price - mean_price) * (total_price - mean_price)) AS var_amount
            FROM (
                SELECT customer_id, total_price, AVG(total_price) AS mean_price
                FROM transactions
                GROUP BY customer_id
            )
            GROUP BY customer_id
        '''
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        columns = ['customer_id', 'avg_var']
        var_df = pd.DataFrame(result,columns = columns)
        return var_df
    
    def category_distribution_by_customers(self):
        query = '''
            SELECT t.customer_id, p.category, COUNT(td.product_id) AS category_count
            FROM transactions t
            JOIN transactions_details td ON t.transaction_id = td.transaction_id
            JOIN products p ON td.product_id= p."Product ID"
            GROUP BY t.customer_id, p.category
            ORDER BY t.customer_id, category_count DESC
        '''
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        
         # Convert the fetched results into a DataFrame
        df = pd.DataFrame(result, columns=['customer_id', 'category', 'category_count'])
        total_products= df.groupby('customer_id')['category_count'].sum().reset_index()
    
        # Pivot the DataFrame to get customer IDs as index and categories as columns with counts
        pivot_df = df.pivot(index='customer_id', columns='category', values='category_count').fillna(0).astype(int)
        
        final_df = pd.merge(pivot_df, total_products, on='customer_id')
        final_df = final_df.rename(columns = {'category_count':'Products Purchased'})
        return final_df
    
    
    def payment_distribution_by_customers(self):
        query = '''
            SELECT t.customer_id, t.payment AS payment_category, COUNT(*) AS payment_count
            FROM transactions t
            GROUP BY t.customer_id, payment_category
            ORDER BY t.customer_id, payment_count DESC
        '''
        self.cursor.execute(query)
        result = self.cursor.fetchall()

        # Convert the fetched results into a DataFrame
        df = pd.DataFrame(result, columns=['customer_id', 'payment', 'payment_count'])
        total_payments = df.groupby('customer_id')['payment_count'].sum().reset_index()

        # Pivot the DataFrame to get customer IDs as index and payment categories as columns with counts
        pivot_df = df.pivot(index='customer_id', columns='payment', values='payment_count').fillna(0).astype(int)

        final_df = pd.merge(pivot_df, total_payments, on='customer_id')
        final_df = final_df.rename(columns={'payment': 'Total Payments'})
        return final_df
    
    
    def avg_price_per_product(self):
        
        query = '''
            SELECT t.customer_id, AVG(p.price) AS average_price
            FROM transactions t
            JOIN transactions_details td ON t.transaction_id = td.transaction_id
            JOIN products p ON td.product_id = p."Product ID"
            GROUP BY t.customer_id;
        '''
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        
        # convert to fetched results into a DataFrame
        df = pd.DataFrame(result, columns=['customer_id','avg_price'])
        return df       
    
    def product_distribution_by_customers(self):
        query = '''
            SELECT t.customer_id, 
                   SUM(CASE WHEN p."On Sale" = 1 AND p."On Display" = 1 THEN 1 ELSE 0 END) AS both_on_sale_and_display,
                   SUM(CASE WHEN p."On Sale" = 1 AND p."On Display" = 0 THEN 1 ELSE 0 END) AS on_sale_only,
                   SUM(CASE WHEN p."On Sale" = 0 AND p."On Display" = 1 THEN 1 ELSE 0 END) AS on_display_only,
                   SUM(CASE WHEN p."On Sale" = 0 AND p."On Display" = 0 THEN 1 ELSE 0 END) AS neither_on_sale_nor_display
            FROM transactions t
            JOIN transactions_details td ON t.transaction_id = td.transaction_id
            JOIN products p ON td.product_id = p."Product ID"
            GROUP BY t.customer_id
            '''
        self.cursor.execute(query)
        result = self.cursor.fetchall()

        # Convert the fetched results into a DataFrame
        df = pd.DataFrame(result, columns=[
            'customer_id', 'both_on_sale_and_display', 'on_sale_only', 'on_display_only', 'neither_on_sale_nor_display'
        ])
        return df

    def merge_features(self):
        # get custoemr data
        # get all the feature data frames
        customer_data = self.get_customer_data()
        purchase_freq_data = self.purchase_frequency()
        total_spent_by_customers = self.total_spent_by_customers()
        avg_cost_by_customers = self.avg_cost_by_customers()
        avg_var_by_customers = self.avg_var_by_customers()
        category = self.category_distribution_by_customers()
        payments = self.payment_distribution_by_customers()
        avg_price = self.avg_price_per_product()
        product_sale_display = self.product_distribution_by_customers()
        # Merge customer data with purchase frequency data based on customer_id
        self.features = pd.merge(customer_data, purchase_freq_data, on='customer_id', how='left')
        
        # Merge additional DataFrame with existing features DataFrame based on customer_id
        self.features = pd.merge(self.features, total_spent_by_customers, on='customer_id', how='left')
        self.features = pd.merge(self.features, avg_cost_by_customers, on='customer_id', how='left')
        self.features = pd.merge(self.features, avg_var_by_customers, on='customer_id', how='left')
        self.features = pd.merge(self.features, category, on='customer_id', how='left')
        self.features = pd.merge(self.features, payments, on='customer_id',how='left')
        self.features = pd.merge(self.features, avg_price, on = 'customer_id', how = 'left')
        self.features = pd.merge(self.features, product_sale_display, on = 'customer_id',how = 'left')
        self.features['buyer_habit_spend']= self.features['buyer_habit']+' '+self.features['buyer_spending']
        self.features = pd.get_dummies(self.features, columns = ['gender'],drop_first = True)
        self.features = self.features.dropna(axis=0, how = 'any')
        save_features_to_csv(self.features)
        return self.features 
    
    def total_sales_by_segment(self):
        df = self.features
        result = df.groupby(['buyer_habit','buyer_spending'])['total_spent'].sum().reset_index()
        pivot_table = result.pivot('buyer_habit','buyer_spending','total_spent')
        
        # Creating a heatmap using seaborn
        plt.figure(figsize=(8, 6))
        colors = ['#FF0000', '#FFFFFF', '#0000FF']  # Red -> White -> Blue
        cmap_name = 'custom_diverging_map'
        n_bins = 1000  # Number of bins for interpolation
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        sns.heatmap(pivot_table, annot=True, cmap=custom_cmap, fmt='.2f', linewidths=0.5)
        plt.title('Total Spent by Buyer Habit and Spending')
        plt.xlabel('Buyer Spending')
        plt.ylabel('Buyer Habit')
        return plt.show()
    
    def scatter_with_spending(self, feature_column_name):
        
        # Create a scatterplot using Seaborn
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=feature_column_name, y='total_spent', data=self.features, alpha=0.5)
        plt.title(f'Scatterplot of {feature_column_name.title()} and Total Spent')
        plt.xlabel(f'{feature_column_name.title()}')
        plt.ylabel('Total Spent')
        plt.grid(True)
        return plt.show()
    
    
def get_most_recent_cust_features(directory='Data\\Simulations\\Customer_Features'):
    
    """
    Retrieves the most recent customer features file from the specified directory.

    Args:
    - directory (str): Directory path where customer feature files are stored (default: 'Data\\Simulations\\Customer_Features').

    Returns:
    - cust_feat_df (DataFrame): DataFrame containing the data from the most recent customer features file.
    """
    
    # Find files matching the pattern for cust_features---
    cust_feat_files = glob.glob(os.path.join(directory, 'cust_features*.csv'))
    if cust_feat_files:
        most_recent_files= max(cust_feat_files, key=os.path.getctime)

    cust_feat_df = pd.read_csv(most_recent_files)
    return cust_feat_df    
    
"""
Performs machine learning tasks on customer features data.

Attributes:
- features_df_file (str): File path of the customer features DataFrame.
- directory (str): Directory path where customer feature files are stored.
- test_size (float): Size of the test dataset (default: 0.2).

Methods:
- __init__(): Initializes the PerformLearning class.
- train_test_split_data(): Splits the data into training and testing sets.
- random_forest(): Performs Random Forest classification.
- logistic_regression(): Performs Logistic Regression classification.
- knn(): Performs K-Nearest Neighbors classification.
- gaussian_naive_bayes(): Performs Gaussian Naive Bayes classification.
- support_vector_machines(): Performs Support Vector Machines classification.
- train_and_evaluate_models(): Trains and evaluates different models.
"""
class PerformLearning:

    def __init__(self, 
                 features_df_file=None,
                 directory='Data\\Simulations\\Customer_Features',
                test_size=0.2):
        """
        Initializes the PerformLearning class.

        Args:
        - features_df_file (str): File path of the customer features DataFrame (default: None).
        - directory (str): Directory path where customer feature files are stored (default: 'Data\\Simulations\\Customer_Features').
        - test_size (float): Size of the test dataset (default: 0.2).
        """
        
        if features_df_file is None:
            self.features_df = get_most_recent_cust_features(directory)
        else:
            self.features_df = pd.read_csv(features_df_file)
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split_data()
        self.train_and_evaluate_models()
    
    def train_test_split_data(self):
        """
        Splits the data into training and testing sets.

        Returns:
        - X_train, X_test, y_train, y_test: Training and testing data.
        """
        # Merge features and labels (assuming labels are already in your features DataFrame)
        # Assuming 'target_label' is your target column in the features DataFrame
        X = self.features_df[['age', 'gender_Male','gender_Non-Binary','gender_Prefer Not To Say','zip',
                              'purchase_count','Products Purchased','total_spent','avg_spent','avg_var', 'avg_price',
                           'Accessories','Beauty','Clothing','Electronics', 
                           'Groceries', 'Home', 'Office Supplies',
                           'Personal Care','Shoes', 
                           'both_on_sale_and_display','on_sale_only','on_display_only','neither_on_sale_nor_display',                       
                           ]] # Features
        
        y = self.features_df['buyer_habit_spend']  # Target

        # Perform train-test split using stratified random sampling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, stratify=y, random_state=42)

        return X_train, X_test, y_train, y_test
    

    def random_forest(self):
        """
        Performs Random Forest classification.

        Returns:
        - Dictionary with the model accuracy and the best model.
        """
        # Define the parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],  # Number of trees in the forest
            'max_depth': [None, 5, 10, 20],  # Maximum depth of the trees
            # Add other hyperparameters to tune
        }

        # Initialize and train the model
        model = RandomForestClassifier(random_state=42)

        # Initialize GridSearchCV with the RandomForestClassifier and parameter grid
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

        # Perform grid search on your data
        grid_search.fit(self.X_train, self.y_train)

        # Get the best parameters and best score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Get the best random forest model from grid search
        best_forest_model = grid_search.best_estimator_

        # Evaluate the best Logistic Regression model on the test set
        accuracy_test = best_forest_model.score(self.X_test, self.y_test)

        # Print the best parameters and best score
        print("Best Parameters:", best_params)
        print("Best Score:", best_score)
        print("Accuracy on Test Set (Random Forest):", accuracy_test)
        return {'random forest': accuracy_test,
               'model':best_forest_model}
    def logistic_regression(self):
        """
        Performs Logistic Regression classification.

        Returns:
        - Dictionary with the model accuracy and the best model.
        """
        # Define the parameter grid for Logistic Regression
        param_grid_logreg = {
            'C': [0.1, 1, 10],              # Regularization parameter
            'solver': ['liblinear', 'saga'] # Optimization algorithm
            # Add other hyperparameters for tuning
        }

        # Initialize the Logistic Regression model
        logistic_reg = LogisticRegression(random_state=42, max_iter = 10000)

        # Initialize GridSearchCV with the Logistic Regression and parameter grid
        grid_search_logreg = GridSearchCV(estimator=logistic_reg, param_grid=param_grid_logreg, cv=5, scoring='accuracy')

        # Perform grid search on your data
        grid_search_logreg.fit(self.X_train, self.y_train)

        # Get the best parameters and best score
        best_params_logreg = grid_search_logreg.best_params_
        best_score_logreg = grid_search_logreg.best_score_

        best_logreg_model = grid_search_logreg.best_estimator_

        # Evaluate the best Logistic Regression model on the test set
        accuracy_logreg_test = best_logreg_model.score(self.X_test, self.y_test)

        # Print the best parameters and best score
        print("Best Parameters (Logistic Regression):", best_params_logreg)
        print("Best Score (Logistic Regression):", best_score_logreg)
        print("Accuracy on Test Set (Logistic Regression):", accuracy_logreg_test)
        return {'logistic regression': accuracy_logreg_test,
               'model':best_logreg_model}

    def knn(self):
        """
        Performs K-Nearest Neighbors classification.

        Returns:
        - Dictionary with the model accuracy and the best model.
        """
        # Define the parameter grid for KNN
        param_grid_knn = {
            'n_neighbors': [3, 5, 7],          # Number of neighbors
            'weights': ['uniform', 'distance'] # Weighting method
            # Add other hyperparameters for tuning
        }

        # Initialize the KNN classifier
        knn = KNeighborsClassifier()

        # Initialize GridSearchCV with the KNN classifier and parameter grid
        grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=5, scoring='accuracy')

        # Perform grid search on your data
        grid_search_knn.fit(self.X_train, self.y_train)

        # Get the best parameters and best score
        best_params_knn = grid_search_knn.best_params_
        best_score_knn = grid_search_knn.best_score_


        best_model = grid_search_knn.best_estimator_
        accuracy_test = best_model.score(self.X_test, self.y_test)

        # Print the best parameters and best score
        print("Best Parameters (KNN):", best_params_knn)
        print("Best Score (KNN):", best_score_knn)
        print("Accuracy on Test Set (KNN):", accuracy_test)
        return {'knn': accuracy_test,
                   'model':best_model}

    def gaussian_naive_bayes(self):
        """
        Performs Gaussian Naive Bayes classification.

        Returns:
        - Dictionary with the model accuracy and the model itself.
        """
        # There are no hyperparameters to tune in Gaussian Naive Bayes
        # Initialize the Gaussian Naive Bayes classifier
        nb = GaussianNB()
        # No parameter grid for Gaussian Naive Bayes as there are no hyperparameters to tune

        # Perform cross-validation on your data
        nb.fit(self.X_train, self.y_train)

        # Evaluate the model (optional if you want to check the performance, as there are no hyperparameters to optimize)
        accuracy_nb = nb.score(self.X_test, self.y_test)
        print("Test Accuracy (Gaussian Naive Bayes):", accuracy_nb)

        return {'gaussian naive bayes': accuracy_nb,
                'model':nb}

    def support_vector_machines(self):
        """
        Performs Support Vector Machines classification.

        Returns:
        - Dictionary with the model accuracy and the best model.
        """
        # Define the parameter grid for SVM
        param_grid_svm = {
            'C': [0.1, 1, 10],          # Regularization parameter
            'kernel': ['linear', 'rbf'] # Kernel type
            # Add other hyperparameters for tuning
        }

        # Initialize the SVM classifier
        svm = SVC(random_state=42)

        # Initialize GridSearchCV with the SVM classifier and parameter grid
        grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, scoring='accuracy')

        # Perform grid search on your data
        grid_search_svm.fit(self.X_train, self.y_train)


        # Get the best parameters and best score
        best_params_svm = grid_search_svm.best_params_
        best_score_svm = grid_search_svm.best_score_

        # get test scores
        best_svm_model = grid_search_svm.best_estimator_
        accuracy_svm_test = best_svm_model.score(self.X_test,self.y_test)
        # Print the best parameters and best score
        print("Best Parameters (SVM):", best_params_svm)
        print("Best Score (SVM):", best_score_svm)
        print("Accuracy on Test Set (SVM):", accuracy_svm_test)
        return {'support vector machine': accuracy_svm_test,
                'model':best_svm_model}
    
    def train_and_evaluate_models(self):
        """
        Trains and evaluates different models.

        Returns:
        - Results of different models including their accuracy and best models.
        """
        # Train and evaluate different models
        rf_result = self.random_forest()
        lr_result = self.logistic_regression()
        knn_result = self.knn()
        nb_result = self.gaussian_naive_bayes()
        #svm_result = self.support_vector_machines()

        return rf_result, lr_result, knn_result, nb_result#, svm_result
    
    
    
