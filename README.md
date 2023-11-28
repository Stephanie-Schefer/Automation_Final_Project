# Automation Final Project
## Stephanie Schefer
## Fall 2023

---------------------
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


## Overall Project Vision

- simulate data
    - customers and their categories and identifying information
    - products and their categories
    - sales transactions
- chart pops up with product category distributions
- machine learning to identify impulse purchases

## 
---------

## Project Plan
  As an entrepreneur and business analytics student, I want to analyze sales data to gain insights into consumer behaviors. I will apply the project workflow and documentation lessons taught in the Automation & Workflows class. Working towards this, I will be able to demonstrate findings to a wide audience which may include business analysts, data scientists, entrepreneurs, academics, students, and more. My analysis will be captured and displayed through publishing a website.

  I am also particularly interested in discussing consumer behavior and preferences among female (large-spender) digital natives in online retail spaces. There is a term "girl math" that refers to applying fuzzy and funny logic to rationalize why they are getting a good deal as far as their time, money, and convenience are concerned, compared to those who think or act differently from them. An example includes buying an expensive accessory but justifying it by putting a per day dollar amount after using such an item frequently over a certain period of time, such as a month or a year. 

# Analytics will include the following

- Customer Segmentation and Targeted Marketing
    - Cluster Analysis
    - Comparative Analysis
- Customer Churn Prediction and Retention
- Predictive Analytics for Sales
    - Demand Forecasting
    
# Steps
#### 1. Data Cleaning and Preparation:
Handle Missing Data: Check for missing values in columns and decide on an appropriate strategy to handle them, such as imputation or removal.
Data Formatting: Ensure consistency in data formats, especially for dates and categorical variables.
Feature Engineering: Create new features from existing ones if needed. For example, can calculate the average transaction amount per age group or segment.
#### 2. Exploratory Data Analysis (EDA):
Age and Gender Analysis: Understand the age and gender distribution of the customers. Visualize this data to identify the primary demographic.
State-wise Analysis: Explore sales patterns based on different states. Identify regions with high and low sales.
Payment Method Analysis: Analyze which payment methods are most popular among different demographic groups.
Segment Analysis: Examine sales performance based on customer segments. Identify which segments contribute the most to revenue.
#### 3. Customer Segmentation:
Use techniques like clustering (K-means, hierarchical clustering) to segment customers based on their purchasing behavior.
Analyze the spending patterns of different segments and tailor marketing strategies accordingly.
#### 4. Consumer Behavior Analysis:
Purchase Frequency: Analyze how often customers make purchases. Identify patterns, such as peak shopping days or months.
Amount Spent Analysis: Investigate the distribution of amounts spent. Identify high spenders and analyze their characteristics and purchasing behaviors.
Correlation Analysis: Explore correlations between variables. For instance, check if there's a correlation between age and amount spent.
#### 5. Predictive Analytics:
Churn Prediction: Predict which customers are likely to churn based on their transaction history and demographics.
Sales Forecasting: Use historical transaction data to forecast future sales. Time series forecasting methods like ARIMA or machine learning algorithms can be applied.
#### 6. Marketing Strategies:
Referral Analysis: Determine the effectiveness of referral programs. Analyze which referral methods bring in the most valuable customers.
Promotion Impact: Analyze the impact of promotions on sales. Identify which promotions result in increased sales and customer engagement.
#### 7. Visualization and Reporting:
Create interactive visualizations and dashboards to present the findings effectively.
Prepare a summary report highlighting key insights, trends, and recommendations for business improvement.
#### 8. Advanced Analysis:
Predictive Modeling: Utilize machine learning algorithms (like Random Forest, Gradient Boosting, or Neural Networks) to predict customer behavior or forecast sales more accurately.

---------------- 
# Data Simulation
## Database Structure

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
- **Type (Sale or Refund)**
- Store ID
- Customer ID
- Purchase Total Amount
- **Payment Method (Credit Card, Debit Card, Gift Card, Cash)**
- Coupon/Promotion Code In Order

##### Sales Transaction Details
- Transaction ID
- **Product ID**
- Quantity Purchased
- **CouponCode**

##### Product Details
- Product ID
- Product Description
- Product Category (electronics, clothing, groceries)
- Price
