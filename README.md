<em>In this repository, I share some of the work I've done as part of the MSc. Big Data and Business Analytics course at IE.</em> 

## 🌟 **1. Understanding the ML Pipeline: Titanic Dataset**

Understanding the main aspects of the practical implementation of a Machine Learning Pipeline, from the data reading to the model training using the Titanic dataset

## 🌟 **2. PySpark MLlib: Titanic Dataset**

Similar to **(1)** but using  <em>MLlib</em> in PySpark. The processing is done on stored data in Hadoop. Also, features are assembled into an ML pipeline.

## **3. Prediction Model: Breast Cancer Detection**

Loading the breast_cancer dataset in sklearn.datasets.load_breast_cancer and playing with different classification models to get the best possible cancer estimator
- Logistic Regression
- Decision Tree
- K-Nearest Neighbor

## **4. Regression Model: Boston Housing**

To predict the median value of owner-occupied homes in Boston

```
Variables in order:
 CRIM     per capita crime rate by town
 ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
 INDUS    proportion of non-retail business acres per town
 CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
 NOX      nitric oxides concentration (parts per 10 million)
 RM       average number of rooms per dwelling
 AGE      proportion of owner-occupied units built prior to 1940
 DIS      weighted distances to five Boston employment centres
 RAD      index of accessibility to radial highways
 TAX      full-value property-tax rate per $10,000
 PTRATIO  pupil-teacher ratio by town
 B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
 LSTAT    % lower status of the population
 MEDV     Median value of owner-occupied homes in $1000's  (THIS IS THE TARGET)
```

- Linear Regression
- Decision Tree

## 🌟 **5. Prediction Model: Bank Marketing**

The dataset collected is related to 17 campaigns that occurred between May 2008 and November 2010, corresponding to a total of 79354 contacts. During these phone campaigns, an attractive long-term deposit application, with good interest rates, was offered. For each contact, a large number of attributes was stored and if there was a success (the target variable). For the whole database considered, there were 6499 successes (8% success rate). 

The dataset can be found in Kaggle (https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset).
