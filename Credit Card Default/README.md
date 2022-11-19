# Credit Card Default
Here, I share my winning solution to the challenge of building a model to predict credit card defaulters accurately **in order to minimize money loss**.

For this task, I was given a set of data on default payments and demographic data to help us do our task. Data is comprised in the following CSV files, each containing a set of information related to 20,000 customers.

## **About the data**

**TRAINING**

**`train_customers.csv`**
 - `ID`: ID of each client
 - `LIMIT_BAL`: Amount of given credit in NT dollars (includes individual and family/supplementary credit)
 - `SEX`: Gender (1=male, 2=female)
 - `EDUCATION`: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
 - `MARRIAGE`: Marital status (1=married, 2=single, 3=others)
 - `AGE`: Age in years
 
**`train_series.csv`**
 - `ID`: ID of each client
 - `MONTH`: The month to which the data is referring to
 - `PAY`: Repayment status in the corresponding month (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
   
   **Updated definition of repayment status**
    - -2: No need to pay, zero balance
    - -1: Paid in full
    - 0: The use of revolving credit i.e the client paid more than the minimum amount due but less than the full amount due
    - 8: Delay of 8 months or more
 - `BILL_AMT`: Amount of bill statement in the corresponding month (NT dollar)
 - `PAY_AMT`: Amount of previous payment in the corresponding month (NT dollar)
 
**`train_target.csv`**
 - `DEFAULT_JULY`: Default payment in July (1=yes, 0=no)
 
 
**TEST**
 - **`test_data.csv`**
 
**SUBMISSION**
 - **`submission_features.csv`**

## **Challenge: Threshold Optimization**

Now the bank wants to optimize the decision-making process by establishing the optimal threshold for the model in order to effectively take the decision about when to issue the credit and when not. So, taking into account the following numbers:

- A customer who received a loan but doesn't repay costs 5000$ to the bank
- A customer who receives a loan and repays, make a profit of 1000$ to the bank
- If the credit is not issued, then there is no profit or loss
