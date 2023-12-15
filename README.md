# Automation Final Project
## Stephanie Schefer
## Fall 2023


---------------------
## Course Description
COLL 400 Designation:
This is a COLL 400 course. It is suitable for the Data Application and Algorithms tracks. Specifically, a COLL 400 course:
Will require students to take the initiative in synthesis and critical analysis, to solve
problems in an applied and/or academic setting, to create original material or original scholarship, and to communicate effectively with a diversity of audiences.
In this capstone course, you will synthesize and apply what you have learned from the Data
Science curriculum as well as the broader COLL curriculum to:
  - Develop your own solution to a problem that you will define.
  - Assist others in solving their problems by testing their code and providing feedback.
  - Build a library of code along with supporting documentation so that a user without
  detailed knowledge of how the code works can still use it. This will be published on
  GitHub.
  - Present your results to the class, keeping your presentation at a level appropriate for a general (i.e., non-expert audience). You will receive feedback on this presentation to
  - prepare you for:
  - Present your results to a broader audience. Options may include:
    – Publishing a webpage
    – Delivering a talk or poster presentation to a larger audience (I and the DS faculty
    can help you coordinate this)
    – I’m open to other suggestions!
---------------------
## Project Inspiration

  As an entrepreneur and business analytics student, I want to analyze sales data to gain insights into consumer behaviors. I will apply the project workflow and documentation lessons taught in the Automation & Workflows class. Working towards this, I will be able to demonstrate findings to a wide audience which may include business analysts, data scientists, entrepreneurs, academics, students, and more. My analysis will be captured and displayed through a notebook file interface.

  I am also particularly interested in discussing consumer behavior and preferences among female (large-spender) digital natives in online retail spaces. There is a term "girl math" that refers to applying fuzzy and funny logic to rationalize why they are getting a good deal as far as their time, money, and convenience are concerned, compared to those who think or act differently from them. An example includes buying an expensive accessory but justifying it by putting a per day dollar amount after using such an item frequently over a certain period of time, such as a month or a year. Thus, gender is one component that will be encorporated into the machine learning models. 
  
## Project Overview

This repository contains the necessary files for generating simulation data. Simulation data is comprised of customers and their associated characteristics, store products and their details, as well as sales transaction data. Simulations are automatically saved for future usage.

Following the data generation simulation, machine learning is applied to identify customer segments. There are 6 segments from the various combinations of impulse or routine spending habits as well as low, medium, or high buyer spending. 

## Results and Analysis For Customer Segmentation
  
Running the simulation with the following specifications resulted in 169367 transactions throughout the year from January 1st, 2022 until December 31, 2022. 
- Simulation
    - num_periods = 365,
    - start_year=2022,
    - start_month = 1,
    - start_day = 1,
    - del_previous_sims = False,
    - cust_prod = None (Generates a CustProd Instance with default settings such as 1000 customers and 300 products)

Running Machine Learning Classification Techniques to identify the various combinations of buyer spending and buying habits using the default test size of 20% resulted in the same accuracy for Random Forest (max_depth none, n_estimators:200), Logistic Regression (C:.1, solver:saga), SVM (C:.1, kernel:rbf). All of these classification methods resulted in 57.5% accuracy on the training set and 60% accuracy on the testing set.

-------
## Workflow For Running Simulation
### Working with the RunningSim.ipynb file....

1. (optional) Customers and Products
    - Function: Generate the customers and products that are in our store. 
    - Parameters: Select the number of customers (int), number of products(int), categories of the products(list of strings), mean of product categories (list of int), standard deviation of product categories (list of int), sales percent (dec), display percent (dec), random seed (int), data_importer (instance of the DataImporter class)
    - Output: Two dataframes are saved, one for customers and their information and the other for products and their details.
    
2. Simulate Transaction Data
    - Function: Generate transaction and transaction details for a specified number of periods (days)
    - Parameters: number of periods (int), start year (int), start month (int), start day (int), delete previous sim (Boolean or list of int), cust_prod (None or an instance of the Customer Products class if one has been created)
    - Output: Transaction and transaction details dataframes are automatically saved. 
    
3. Machine Learning: identify customer segments
    - Function: Classify customer segments
    - Parameters: features_df_file (None or the path to a specific dataframe of engineered features), directory (optional: link to the folder of engineered features where the most recent will be selected), test size (dec)
    - Logic: 
        - The most recent simulation files are collected by default for analysis. To select a different dataset, simply set the path. 
        - This file is stored into a sqlite database based on the database structure outliend below. 
        - Summary statistics for each customer with transactions are generated based on the transaction data.
            - Number of transactions
            - total spent
            - average transaction cost
            - average variance in transaction cost
            - product category distribution counts
            - number of products purchased
            - payment type distribution counts
            - average price of product purchased
            - number of products purchased on sale and on display, only on sale, only on display, neither on display nor on sale
            - customer features (gender, age, zip,etc.)
        - Machine learning is applied to identify customer segments. 
            - Training and Testing sets of 80% and 20% are used.
            - Multiple methods such as random forest, logistic regression, knn, gaussian naive bayes, and support vector machines are applied to the dataset. Here I use grid search techniques to test out various parameters. The respective performing accuracy is reported. 

## Assumptions for synthetic data generation
1. Customer Attributes:
- Age Distribution: Customers' ages are randomly generated between 18 and 85 using np.random.randint(18, 85, size=num_cust).
- Gender Distribution: Randomly selected genders are assigned to customers based on probabilities specified (['Male', 'Female', 'Non-Binary', 'Prefer Not To Say']) with respective probabilities (0.45,0.45,0.05,0.05).
- Location: Zip codes are randomly selected from a provided list of zip codes (uszips.csv file) to assign locations to customers.
- Buyer Habits: Customers' purchasing habits (impulse or routine) and spending levels (heavy, moderate, or light) are randomly assigned based on given probabilities (habit: 0.25, 0.75)(spending:0.2, 0.7, 0.1).

2. Product Attributes:
- Category and Pricing: Products are categorized into various categories (e.g., Clothing, Shoes) and their prices are randomly generated using normal distributions with provided mean and standard deviation values for each category.
- Sale and Display: Products are randomly marked as being on sale or display based on provided percentages (sale_pct and display_pct).

3. Sales Generation:
- Base Sales: Sales are simulated to increase over time (base_sales) with some random noise.
- Economic Conditions: Random fluctuations in economic conditions are introduced to simulate their impact on sales volume.
- Customer Purchases: Customers are randomly selected to make purchases on specific days. Their purchases depend on their spending habits and the available products. For example, the range of a transaction cost is -50 dollars for a light spender, 50-100 dollars for medium spenders and 100-150 for heavy spenders. Impulse purchasers have a 70% chance of purchasing a product that is either on sale, on display, or both. The products customers purchase are based on their target transaction cost and their likeliness to select products based on thier features. 

4. Transaction Generation:
- Transaction Time: Transactions are generated within a specific period (365 days by default), and each transaction occurs at a random time during the day.
- Payment Method: Payment methods (card, gift card, cash) are randomly chosen for each transaction based on specified probabilities.

5. Saving Generated Data:
- The code saves transaction data and details periodically during the generation process to prevent data loss in case of unexpected interruptions.

6. Simulation File Naming:
- The simulation generates transaction and transaction detail files with naming conventions that include a simulation number and the count of transactions to distinguish between different simulations.


---------------- 
# Further Detail
## Database Structure
The following database is created throughout the project. 

##### Customer Information Data Set
- Customer ID
- First Name
- Last Name
- Age
- Gender (Female, Male, Non-Binary, Prefer Not to Say)
- Zip Code Location
- Buyer Habit (Impulse and Routine)
- Buying Spending (High, Medium, or Low Spender)

##### Sales Transaction Data
- Transaction ID
- Transaction Date Time
- Store ID
- Customer ID
- Purchase Total Amount
- **Payment Method (Credit Card, Debit Card, Gift Card, Cash)**
- Coupon/Promotion Code In Order

##### Sales Transaction Details
- Transaction ID
- **Product ID**
- Product Price
- Product Sale
- Product Display
##### Product Details
- Product ID
- Product Category (electronics, clothing, groceries)
- Price
- **On Display**
- **On Sale**


------------
# Procedures Conducted Throughout the Project
#### 1. Data Preparation:
Data Generation: Simulate sales transaction data for a store, creating the customers, products, and their purchase history. 
Data Storage: Create a database for future recollection. 
Data Formatting: Ensure consistency in data formats, especially for dates and categorical variables.
Feature Engineering: Create new features from existing ones if needed. For example, can calculate the average transaction amount per age group or segment.
#### 2. Exploratory Data Analysis (EDA):
Age and Gender Analysis: Understand the age and gender distribution of the customers. Visualize this data to identify the primary demographic.
Zip-wise Analysis: Distribution of customer's location. Explore sales patterns based on different zip codes. Identify regions with high and low sales.
Segment Analysis: Examine sales performance based on customer segments. Identify which segments contribute the most to revenue.
#### 3. Customer Segmentation:
Use techniques like clustering (K-means, hierarchical clustering) to segment customers based on their purchasing behavior.
Analyze the spending patterns of different segments and tailor marketing strategies accordingly.
#### 4. Consumer Behavior Analysis:
Purchase Frequency: Analyze how often customers make purchases. Identify patterns, such as peak shopping days or months.
Amount Spent Analysis: Investigate the distribution of amounts spent. Identify high spenders and analyze their characteristics and purchasing behaviors.
Correlation Analysis: Explore correlations between variables. For instance, check if there's a correlation between age and amount spent.
#### 5. Visualization and Reporting:
Create interactive visualizations to present the findings effectively.
#### 6. Advanced Analysis:
Predictive Modeling: Utilize machine learning algorithms (like Random Forest, SVM, etc.) to predict customer buying behavior.
