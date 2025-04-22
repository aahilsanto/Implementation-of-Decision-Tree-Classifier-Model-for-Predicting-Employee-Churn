# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: A Ahil Santo

RegisterNumber: 212224040018

```
import pandas as pd
data=pd.read_csv(r"E:\Desktop\CSE\Introduction To Machine Learning\dataset\Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
print(data.head())
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

![1](https://github.com/user-attachments/assets/c232078d-29cf-434f-9588-650ddc65e52f)


![2](https://github.com/user-attachments/assets/ac4c4f05-ac88-4de7-85a3-4bb294cc5cd2)

![3](https://github.com/user-attachments/assets/dc2755f1-ea9e-4f3c-bb1b-9be502b8d08f)

![4](https://github.com/user-attachments/assets/5a1a050e-2c88-4316-b337-54b344cc3b6a)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
