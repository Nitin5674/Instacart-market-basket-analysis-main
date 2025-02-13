# 📊 Instacart Market Basket Analysis

### 🚀 Project Overview

Instacart Market Basket Analysis aims to uncover customer purchasing behavior by analyzing transactional data. This project applies data science techniques to extract meaningful insights from customer orders, helping businesses optimize product recommendations and inventory management.

### 📌 Key Objectives

- Understand purchasing patterns of Instacart customers.

- Identify frequently bought items together using association rule mining.

- Segment customers based on their buying habits.

- Predict future purchases using machine learning models.

### 📂 Dataset

The dataset is sourced from Instacart’s open [data](https://www.kaggle.com/c/instacart-market-basket-analysis/data), containing 3 million grocery orders from 200,000 users. The dataset includes:

- order_products_prior.csv - Prior orders of users

- orders.csv - Order metadata

- products.csv - Product details

- aisles.csv - Aisle details

- departments.csv - Department details

### Project Structure
```
.
├── Plots/                                      : Contains all plots 
├── Data Description and Analysis.ipynb         : Initial analysis to understand data
├── Exploratory Data Analysis.ipynb             : EDA to analyze customer purchase pattern
├── Customers Segmentation.ipynb                : Customer Segmentation based on product aisles
├── Market Basket Analysis.ipynb                : Market Basket Analysis to find products association
├── Feature Extraction.ipynb                    : Feature engineering and extraction for a ML model
├── Data Preparation.ipynb                      : Data preparation for modeling
├── ANN Model.ipynb                             : Neural Network model for product reorder prediction
├── XGBoost Model.ipynb                         : XGBoost model for product reorder prediction
├── LICENSE                                     : License
└── README.md                                   : Project Report
```
<br />
This project structure offers a systematic approach, breaking down tasks into manageable steps, with each notebook focusing on a specific task in the data analysis and modeling pipeline.


## Markest Basket Analysis

Market Basket Analysis is a modelling technique based upon the theory that if you buy a certain group of items, you are more or less likely to buy another group of items. Market basket analysis may provide the retailer with information to understand the purchase behavior of a buyer. 

Market basket analysis scrutinizes the products customers tend to buy together, and uses the information to decide which products should be cross-sold or promoted together. The term arises from the shopping carts supermarket shoppers fill up during a shopping trip.

Association Rule Mining is used when we want to find an association between different objects in a set, find frequent patterns in a transaction database, relational databases or any other information repository.

The most common approach to find these patterns is Market Basket Analysis, which is a key technique used by large retailers like Amazon, Flipkart, etc to analyze customer buying habits by finding associations between the different items that customers place in their “shopping baskets”. 
- Changing the store layout according to trends
- Customers behavior analysis
- Catalog Design
- Cross marketing on online stores
- Customized emails with add-on sales, etc.

### Matrices

**Support** : Its the default popularity of an item. In mathematical terms, the support of item A is the ratio of transactions involving A to the total number of transactions.

**Confidence** : Likelihood that customer who bought both A and B. It is the ratio of the number of transactions involving both A and B and the number of transactions involving B.
- Confidence(A => B) = Support(A, B)/Support(B)

**Lift** : Increase in the sale of A when you sell B.
- Lift(A => B) = Confidence(A, B)/Support(B)
      
- Lift (A => B) = 1 means that there is no correlation within the itemset.
- Lift (A => B) > 1 means that there is a positive correlation within the itemset, i.e., products in the itemset, A, and B, are more likely to be bought together.
- Lift (A => B) < 1 means that there is a negative correlation within the itemset, i.e., products in itemset, A, and B, are unlikely to be bought together.
    
**Apriori Algorithm:** Apriori algorithm assumes that any subset of a frequent itemset must be frequent. Its the algorithm behind Market Basket Analysis. Say, a transaction containing {Grapes, Apple, Mango} also contains {Grapes, Mango}. So, according to the principle of Apriori, if {Grapes, Apple, Mango} is frequent, then {Grapes, Mango} must also be frequent.

I utilized apriori algorithm from Mlxtend python library and found out associations from top 100 most frequent products which resulted in 28 product pairs (total 56 rules) that have lift highr than 1. The top 10 product pairs having highest lift are shown below:

| Product A  | Product B | Lift |
| ------------- | ------------- | ---- |
| Limes  | Large Lemons  | 3 |
| Organic Strawberries | Organic Raspberries | 2.21 |
| Organic Avocado | Large Lemon | 2.12 |
| Organic Strawberries | Organic Blueberries | 2.11 |
| Organic Hass Avocado | Organic Raspberries | 2.08 |
| Banana | Organic Fuji Apple | 1.88 |
| Bag of Organic Bananas | Organic Raspberries | 1.83 |
| Organic Hass Avocado | Bag of Organic Bananas | 1.81 |
| Honeycrisp Apple | Banana | 1.77 |
| Organic Avocado | Organic Baby Spinach | 1.70 |

## ML Model to Predict Product Reorders

We can utilize this anonymized transactional data of customer orders over time to predict which previously purchased products will be in a user’s next order. This would help recommend the products to a user. 

To build a model, I need to extract features from previous order to understand user's purchase pattern and how popular the particular product is. I extract following features from the user's transactional data.

**Product Level Features:** To understand the product's popularity among users
```
(1) Product's average add-to-cart-order
(2) Total times the product was ordered
(3) Total times the product was reordered
(4) Reorder percentage of a product
(5) Total unique users of a product
(6) Is the product Organic?
(7) Percentage of users that buy the product second time
```

**Aisle and Department Level Features:** To capture if a department and aisle are related to day-to-day products (vegetables, fruits, soda, water, etc.) or once-in-a-while products (medicines, personal-care, etc.) 
```
(8) Reorder percentage, Total orders and reorders of a product aisle
(9) Mean and std of aisle add-to-cart-order
(10) Aisle unique users
(10) Reorder percentage, Total orders and reorders of a product department
(11) Mean and std of department add-to-cart-order
(12) Department unique users
(13) Binary encoding of aisle feature (Because one-hot encoding results in many features and make datarame sparse)
(14) Binary encoding of department feature (Because one-hot encoding results in many features and make datarame sparse)
```

**User Level features:** To capture user's purchase pattern and behavior
```
(15) User's average and std day-of-week of order
(16) User's average and std hour-of-day of order
(17) User's average and std days-since-prior-order
(18) Total orders by a user
(19) Total products user has bought
(20) Total unique products user has bought
(21) user's total reordered products
(22) User's overall reorder percentage
(23) Average order size of a user
(24) User's mean of reordered items of all orders
(25) Percentage of reordered itmes in user's last three orders
(26) Total orders in user's last three orders
```

**User-product Level Features:** To capture user's pattern of ordering-reordering specific products 
```
(27) User's avg add-to-cart-order for a product
(28) User's avg days_since_prior_order for a product
(29) User's product total orders, reorders and reorders percentage
(30) User's order number when the product was bought last
(31) User's product purchase history of last three orders
```

### ML Models

Using the extracted features, I prepared a dataframe which shows all the products user has bought previously, user level features, product level features, asile and department level features, user-product level features and the information of current order such as order's day-of-week, hour-of-day, etc. The Traget would be 'reordered' which shows how many of the previously purchased items, user ordered this time. 

Since the dataframe is huge, I reduced the memory consumption of it by downcasting to fit the data int my memory. I preferred MinMaxScaler over StandardScaler as the latter requires 16 GB of RAM for its operation. I followed standard process for model building and I relied on XGBoost as it handles large data, can be parallelized and gives feature importance. I also built Neural Network to see what would be the best performance from this model disregarding some inherent randomness from both of these models.  To balance the data, I have used cost-sensitive learning by assigning class weightage (~{0:1, 1:10}). I have not used random-upsampling/SMOTE as it would increase the data size and I do not have much memory. Also, since random-down-sampling discards information which might be important and would result in bias. 

Since, we can hack the F1 score by changing the threshold, I relied on AUC Score for model evaluation. The performance of both of these models is shown below using Confusion Matrix, ROC curve and classification report. The feature important plot from XGBoost model is also shown to understand important features which help predict product's reorder. The performance of both models is almost similar and XGBoost slightly performs better in terms of ROC-AUC.



## ⚙️ Installation & Setup

Clone the repository and install dependencies:
### Clone this repository
git clone https://github.com/manishdevdi/Instacart-Market-Basket-Analysis.git
cd instacart-market-basket-analysisInstall dependencies

### Install required libraries
[pip install -r requirements.txt](https://github.com/manishdevdi/Instacart-Market-Basket-Analysis/blob/main/requirements.txt)

## 🚀 Future Improvements

- Implement deep learning-based recommendation models.

- Improve data cleaning for better accuracy.

- Analyze seasonal trends in purchases.

- Integrate with real-time transaction data for live recommendations.
  
## 🤝 Contributing

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests to improve the project.

## 📜 License

This project is licensed under the MIT License.


## 🌟 If you found this project useful, don’t forget to star ⭐ the repository!


    
