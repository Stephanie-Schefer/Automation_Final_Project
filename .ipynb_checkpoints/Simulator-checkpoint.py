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
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

class DataImporter:
    def __init__(self, file_directory = 'Data', first = 'GenderNeutralNames.csv',last ='LastNames.csv',zips = 'uszips.csv'):
        self.directory = file_directory
        self.file_first = first
        self.file_last = last
        self.file_zips = zips
        self.first = self.import_first()
        self.last = self.import_last()
        self.zip = self.import_zips()
    
        
    def import_first(self): #process = False
        file_name = fr"{self.directory}/{self.file_first}"
        with open(file_name,'r')as file:
            #if process: 
            # read the names from the file and remove new line characters
            excel_list = [line.strip() for line in file.readlines()]
            names = [row.replace('Â\xa0','').replace(" ","").split('.')[-1] for row in excel_list if any(char.isdigit() for char in row)]
        return names

    def import_last(self):
        file_name = fr"{self.directory}/{self.file_last}"
        # Open the file in read mode
        with open(file_name,'r')as file:
            # read the names from the file and remove new line characters
            excel_list = [line.strip() for line in file.readlines()]

        # Process Names
        last_names = []
        for row in excel_list:
            # remove unwanted characters (Â\xa0, extra spaces, and consecutive commas)
            cleaned_row = row.replace('Â\xa0', '').replace(" ", "").replace(',,', '')

            # Extract the text after the last digit
            if any(char.isdigit() for char in cleaned_row):
                last_digit_idx = max([i for i, char in enumerate(cleaned_row) if char.isdigit()])
                extracted_name = cleaned_row[last_digit_idx+1:]
                # add the name to the last name list
                last_names.append(extracted_name)
        return last_names

    def import_zips(self):
        file_name = fr"{self.directory}/{self.file_zips}"
        # get zip code list
        zips = pd.read_csv(file_name)
        continental_zips_df = zips[~zips['state_name'].isin(['Puerto Rico','Virgin Islands'])]
        # Assuming continental_zips_df is your DataFrame
        continental_zips_df['zip_6']=continental_zips_df['zip']
        continental_zips_df_copy = continental_zips_df.copy()
        continental_zips_df_copy.loc[:, 'zip_6'] = continental_zips_df_copy['zip'].astype(str).str.zfill(5)

 #astype(str).str.zfill(5)
        zip_list = list(continental_zips_df_copy.zip_6.dropna())
        return zip_list   
    
    
#class DataSaver:
 #   @staticmethod
def save_data_to_csv(data, filename_prefix, directory="Data"):
   # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Construct the file path
    file_path = os.path.join(directory, f"{filename_prefix}.csv")

    # Save the data to CSV
    data.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")
   
 # @staticmethod
def delete_sim_files(simulation_versions=None, directory='Data\Simulations'):
    files_to_delete=[]

    # get all files in directory
    all_files = os.listdir(directory)

    # Filter files that match the pattern
    if simulation_versions==None: # delete all files
        for file_name in all_files:
            if file_name.startswith("transactions_details_s") or file_name.startswith("transactions_s"):
                files_to_delete.append(file_name)
    else:  
        for simulation_version in simulation_versions:
            for file_name in all_files:
                if simulation_version:
                    if f"s{simulation_version}_" in file_name:
                        files_to_delete.append(file_name)
                else:
                    if file_name.startswith("transactions_details_s") or file_name.startswith("transactions_s"):
                        files_to_delete.append(file_name)

    # Delete the files
    for file_to_delete in files_to_delete:
        file_path = os.path.join(directory, file_to_delete)
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    return None      

class GenCustProd:
    def __init__(self,
                 #data_importer,
                 num_cust = 1000,
                 num_prod = 300,
                 categories = ['Clothing', 'Shoes','Accessories','Home','Beauty','Personal Care','Office Supplies','Electronics','Groceries'], 
                 means = [30,40,25,60,20,10,10,150,5],
                 st_dev = [10,10,20,30,10,5,5,40,2],
                 sale_pct = .2,
                 display_pct = .1,
                 random_seed = 42,
                 data_importer = None):
        # initialize variables
        #self.data_saver = DataSaver()
        if data_importer == None:
            self.data_importer = DataImporter()
        else:
            self.data_importer = data_importer # allow you to pass in your own instance of customers and products
        self.data_importer = DataImporter()
        self.num_cust = num_cust
        self.num_prod = num_prod
        self.categories = categories
        self.means = means
        self.st_dev = st_dev
        self.sale_pct = sale_pct
        self.display_pct = display_pct
        self.random_seed = random_seed
        self.customers_df = self.gen_customers()
        self.prod_df = self.gen_products()
        self.plot_prod_cat(self.prod_df)
        self.plot_customers_gender(self.customers_df)
        self.plot_customers_age(self.customers_df)
        self.plot_states(self.customers_df)
        
        
    def gen_customers(self):
        # number of unique customers who have made purchases
        # random state for consistent results
        np.random.seed(self.random_seed)
        
        self.first_names = self.data_importer.first
        self.last_names = self.data_importer.last
        self.zip_codes = self.data_importer.zip

        # Generate data for customer information
        customer_data = {
            'Customer_ID': [str(i).zfill(4) for i in np.arange(1,self.num_cust+1)], # 200 customers
            'First_Name': np.random.choice( self.first_names,size=self.num_cust,replace=True),
            'Last_Name': np.random.choice(self.last_names,size=self.num_cust,replace=True),
            'Age': np.random.randint(18,85, size = self.num_cust),
            'Gender': np.random.choice(['Male','Female','Non-Binary','Prefer Not To Say'],
                                       size = self.num_cust,
                                       replace = True,
                                       p= [0.45,0.45,0.05,0.05]),
            'Location': np.random.choice(self.zip_codes, size=self.num_cust, replace=True),
            "Buyer_Habit": np.random.choice(['impulse','routine'],
                                            size = self.num_cust,
                                            replace = True,
                                            p=[0.25, 0.75]),
            "Buyer_Spending": np.random.choice(['heavy','moderate','light'],
                                               size = self.num_cust,
                                               replace = True,
                                               p = [0.2, 0.7, 0.1])
        }

        customers_df = pd.DataFrame(customer_data)
        customers_df['Location']= customers_df['Location'].astype(str).str.zfill(5)
        file_name = "Data/uszips.csv"
        zips_df = pd.read_csv(file_name)
        zips_df_sub = zips_df[['zip','state_name']]
        zips_df_sub_copy = zips_df_sub.copy()
        zips_df_sub_copy['zip'] = zips_df_sub_copy['zip'].astype(str).str.zfill(5)
        
        customers_with_state = pd.merge(customers_df, zips_df_sub_copy, left_on='Location', right_on='zip', how='left')
        customers_with_state.drop('Location', axis =1, inplace = True)
        
        save_data_to_csv(customers_with_state, filename_prefix="customers", directory="Data")
        return customers_with_state
    ## Variables to incorporate later ##
    # Customer Lifetime Value (CLV)
    # Customer Segment (Through analysis: high spenders, occasional buyers, loyal customers)
    # Feedback and Ratings
        
    
    def gen_products(self):
        # set random seed
        np.random.seed(self.random_seed)

        # make product table
        # Generate product IDs
        product_ids = [str(i).zfill(3) for i in range(1, self.num_prod + 1)]

        product_info = {}

        # Convert DataFrame to pricing_distributions dictionary
        pricing_distributions = {}

        for category, mean, st_dev in zip(self.categories, self.means, self.st_dev):
            pricing_distributions[category] = {'mean': mean, 'st_dev': st_dev}

        for prod_id in product_ids:
            category = np.random.choice(self.categories)  # Random category

            # Adjust prices for specific categories
            category_mean = pricing_distributions[category]['mean']
            category_std_dev = pricing_distributions[category]['st_dev']
            price = np.abs(np.round(np.random.normal(category_mean, category_std_dev),2))

            # impulse buying features
            on_sale = np.random.choice([True, False],p = [self.sale_pct,(1-self.sale_pct)])
            on_display = np.random.choice([True,False], p = [self.display_pct,(1-self.display_pct)])

            # add to df
            product_info[prod_id]={
                'Category': category,
                'Price': price,
                'On Sale': on_sale,
                'On Display': on_display
                }

        products_df = pd.DataFrame(product_info).T.reset_index()
        products_df.columns = ['Product ID','Category','Price','On Sale','On Display']
        save_data_to_csv(products_df, filename_prefix="products", directory="Data")
        return products_df
    
    def plot_prod_cat(self, product_info_df):

        category_counts = product_info_df['Category'].value_counts()

        plt.figure(figsize=(8,6))
        category_counts.plot(kind = 'bar', color = 'skyblue', edgecolor = 'black')
        plt.title('Category Distribution')
        plt.ylabel('Frequency')
        # add visual saving the plot
        
        return plt.show()
    
    def plot_customers_gender(self,customer_info_df):
        gender_counts = customer_info_df['Gender'].value_counts()
        plt.figure(figsize=(8,6))
        gender_counts.plot(kind='bar', color = 'skyblue', edgecolor = 'black')
        plt.title('Gender Distribution')
        plt.ylabel('Frequency')
        # add visual saving the plot
        
        return plt.show()
    
    def plot_customers_age(self, customer_info_df):
        age_series = customer_info_df['Age']
        num_bins = 15
        plt.hist(age_series,bins=num_bins, color = 'skyblue', edgecolor='black')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.title('Age Distribution')
        # add visual saving the plot
        
        return plt.show()
    
    def plot_states(self, customer_info_df):
        states = customer_info_df['state_name'].value_counts()
        
        plt.figure(figsize=(8,6))
        states.plot(kind='bar', color = 'skyblue', edgecolor = 'black')
        plt.xlabel('State')
        plt.title('State Distribution')
        return plt.show()



        
class SalesGenerator:
    def __init__(self,
                 #cust_prod_instance,
                 num_periods = 365,
                 start_year=2022,
                 start_month = 1,
                 start_day = 1,
                 del_previous_sims = False,
                 cust_prod = None):
                 
        # initialize variables
        if cust_prod == None:
            self.cust_prod_instance = GenCustProd()
        else:
            self.cust_prod_instance = cust_prod # allow you to pass in your own instance of customers and products
        #self.data_saver = DataSaver()
        self.num_periods = num_periods
        self.start_year = start_year
        self.start_month = start_month
        self.start_day = start_day
        # erase previous sims
        if del_previous_sims is True:
            delete_sim_files()
        elif del_previous_sims is not False:
            if isinstance(del_previous_sims, list):
                delete_sim_files(simulation_versions=del_previous_sims)
            else:
                raise ValueError("Invalid input for del_previous_sims. Pass True, False, or a list of simulation numbers.")
        #self.cust_prod_instance = cust_prod_instance
        self.simulation_number = self.get_most_recent_simulation_number()+1
        self.sales_data = self.gen_sales()
        self.transactions_df = self.sales_data[0]
        self.transactions_details_df = self.sales_data[1]
        
        

    @staticmethod
    def extract_number_from_string(string):
        # Using regular expression to find the number after 's'
        match = re.search(r's(\d+)', string)
        if match:
            return int(match.group(1))
        else:
            return None
    
    def get_most_recent_simulation_number(self, directory="Data\Simulations"):
        files = os.listdir(directory)
        simulation_numbers = []

        for file in files:
            # transactions_details_s1_30000
            if file.startswith("transactions_details"):
                try:
                    sim_num = file.split("_")[2]
                    number = self.extract_number_from_string(sim_num)
                    #print(f"Extracted number: {number}")
                    if number != None:
                        simulation_numbers.append(number)
                except IndexError:
                    pass
        # print(simulation_numbers)
        if simulation_numbers:
            return max(simulation_numbers)
        else:
            return 0
   
    def save_output_with_sim_number(self, transactions, transaction_details, simulation_number):
        transactions_df = pd.DataFrame(transactions)
        transaction_details_df = pd.DataFrame(transaction_details)

        transactions_filename = f"transactions_s{simulation_number}_{len(transactions_df)}"
        transactions_details_filename = f"transactions_details_s{simulation_number}_{len(transaction_details_df)}"

        save_data_to_csv(transactions_df, filename_prefix=transactions_filename, directory="Data\Simulations")
        save_data_to_csv(transaction_details_df, filename_prefix=transactions_details_filename, directory="Data\Simulations")
        print('saved files')
        return None
    
    def gen_sales(self):
        start_date = date(self.start_year,self.start_month,self.start_day)
        rng = pd.date_range(start_date, periods=self.num_periods, freq='D')
        
        #end_date = self.start_date + timedelta(days=self.num_periods-1)

        # Generate time periods
        time_periods = np.arange(1, self.num_periods + 1)

        # Simulate increasing sales over time with random noise
        base_sales = 100 + 2 * time_periods + np.random.normal(scale=100, size=self.num_periods)

        # Simulate economic conditions fluctuations
        economic_conditions = np.random.uniform(low=0.5, high=1.5, size=self.num_periods)

        #plt.plot(base_sales)
        #simulation_number = self.get_most_recent_simulation_number()+1        
        transactions = []
        transaction_details = []
        saving_incrimentor = 1
        
        for single_date in rng:
            print(single_date)
            d1 = date(single_date.year, single_date.month, single_date.day)
            period = (d1 - start_date).days +1
            
            #periods = delta.days

            # Simulate increasing sales with random noise
            noise = np.random.uniform(0.9, 1.1)

            sales_count = np.abs(int(base_sales[period - 1] * (1 + np.random.uniform(-0.2, 0.2)) * economic_conditions[period - 1]))
            # Adjust the base sales count as needed

            # Randomly select customers who make purchases on this day
            # Customers can make multiple transactions
            customers_purchased = self.cust_prod_instance.customers_df.sample(n=sales_count, replace=True)

            for index, customer in customers_purchased.iterrows():
                # Determine the target amount based on buyer spending habits
                if customer['Buyer_Spending'] == 'light':
                    target_sale = np.random.uniform(5, 50)
                elif customer['Buyer_Spending'] == 'moderate':
                    target_sale = np.random.uniform(50, 100)
                else:
                    target_sale = np.random.uniform(100, 150)

                # Determine products to purchase based on buying habit
                # Based on available products
                available_products = self.cust_prod_instance.prod_df.copy()
                if customer['Buyer_Habit'] == 'Impulse':
                    # Introduce a higher likelihood for products on sale or display
                    impulse_choice_prob = 0.7  # Adjust as needed
                    impulse_products = available_products[
                        (available_products['On_Sale'] == True) | 
                        (available_products['On_Display'] == True)
                    ]
                    non_impulse_products = available_products.drop(impulse_products.index)

                    # Randomly select from impulse and non-impulse products based on probability
                    impulse_product_count = int(impulse_choice_prob * len(impulse_products))
                    selected_impulse_products = impulse_products.sample(n=impulse_product_count, replace=True)
                    selected_non_impulse_products = non_impulse_products.sample(n=(len(available_products) - impulse_product_count), replace=True)

                    available_products = pd.concat([selected_impulse_products, selected_non_impulse_products])

                # selecting products purchased of the available
                purchased_products = []
                purchased_category = []
                purchased_price = []
                purchased_sale = []
                purchased_display = []

                total_price = 0.0

                while target_sale > 0 and not available_products.empty:
                    product = available_products.sample(1)
                    product_price = product['Price'].values[0]

                    if product_price <= target_sale:
                        purchased_products.append(product['Product ID'].values[0])
                        purchased_price.append(product['Price'].values[0])
                        purchased_category.append(product['Category'].values[0])
                        purchased_sale.append(product['On Sale'].values[0])
                        purchased_display.append(product['On Display'].values[0])
                        total_price += product['Price'].values[0]
                        target_sale -= product['Price'].values[0]
                        available_products = available_products.drop(product.index)
                    else:
                        available_products = available_products.drop(product.index)

                # Create timestamp for the transaction within the day
                sale_time = single_date + timedelta(seconds=np.random.randint(86400))  # 86400 seconds in a day
                transaction_id = str(uuid.uuid4())
                payment = np.random.choice(['card','gift card','cash'], p=[0.75,0.1,0.15])
                
                

                # create a transaction record
                transaction = {
                    'transaction_id': transaction_id,
                    'customer_id': customer['Customer_ID'],
                    'purchase_time': sale_time,
                    'total_price': total_price,
                    'payment': payment
                }

                # add transaction to list of transactions
                transactions.append(transaction)

                # create transaction detail record
                for product in range(len(purchased_products)):

                    transaction_detail = {
                        'transaction_id': transaction_id,
                        'product_id': purchased_products[product],
                        'product_price': purchased_price[product],
                        'product_sale': purchased_sale[product],
                        'product_display': purchased_display[product]
                    }
                    transaction_details.append(transaction_detail)
                    
                # save temp file
                if len(transactions)>= 1000*saving_incrimentor:
                    self.save_output_with_sim_number(transactions,transaction_details, self.simulation_number)
                    saving_incrimentor +=1
                
        # save final file
        # Convert transactions data to a DataFrame
        #ransactions_df = pd.DataFrame(transactions)
        # save transactions_df
        #self.data_saver.save_data_to_csv(transactions_df, filename_prefix="transactions", directory="Data")
        self.save_output_with_sim_number(transactions,transaction_details, self.simulation_number)
                   

        # Convert transactions_detail data to a DataFrame
        #transactions_detail_df = pd.DataFrame(transaction_details)
        # save transactions_details_df
        #self.data_saver.save_data_to_csv(transactions_detail_df, filename_prefix="transactions_details", directory="Data")
        
        return 'Finished Sim'#transactions_df, transactions_detail_df