# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
from google.colab import drive
drive.mount('/content/drive')

ls drive/MyDrive/'Colab Notebooks'/DATA/

# **Exploratory data analysis**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
%matplotlib inline
cf.go_offline()

titan=pd.read_csv('drive/MyDrive/Data Science/titanic_dataset.csv')

titan.head()

![image](https://github.com/user-attachments/assets/1770b92f-689a-4220-8cbf-0bfd163c551b)

titan.isnull()

![image](https://github.com/user-attachments/assets/57ca1eb8-dccb-4a96-93f2-7563dcebd2b1)

sns.heatmap(titan.isnull(),yticklabels=False,cbar=False,cmap = 'viridis')

![image](https://github.com/user-attachments/assets/2079a56e-4257-4c03-8823-113bc00904cc)

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=titan,palette='RdBu_r')

![image](https://github.com/user-attachments/assets/f55fe3f4-5d55-4d14-8b8a-5d2412ccac6b)

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=titan,palette='RdBu_r')

![image](https://github.com/user-attachments/assets/e219d497-3c3f-4d23-a710-9bf110fd8fe2)

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=titan,palette='rainbow')

![image](https://github.com/user-attachments/assets/26ec2c9a-5594-4833-b92c-dd7730593c3d)

sns.displot(titan['Age'].dropna(),kde=False,color='darkred',bins=40)
 
![image](https://github.com/user-attachments/assets/a6ade468-b970-4c81-8f0e-5474c32a3732)


titan['Age'].hist(bins=30,alpha=0.3)

![image](https://github.com/user-attachments/assets/cf71bdaa-6043-4408-81ed-ffdd145017b7)

sns.countplot(x='SibSp',data=titan)
 
 ![image](https://github.com/user-attachments/assets/fdd0d843-f68a-4246-aa02-6ce2066b8d33)

titan['Fare'].hist()

![image](https://github.com/user-attachments/assets/54b36fa6-4f2c-4d5a-9a8a-bd891175c99e)

plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=titan,palette='winter')

![image](https://github.com/user-attachments/assets/f895ba19-d181-4de8-a5d0-b43f3f02a9ee)

def impute_age(cols):
  Age=cols[0]
  Pclass=cols[1]
  if pd.isnull(Age):
    if Pclass == 1:
      return 37
    elif Pclass == 2:
      return 29
    else:
      return 24
  else:
    return Age

titan['Age'] = titan[['Age','Pclass']].apply(impute_age,axis=1)

sns.heatmap(titan.isnull(),yticklabels=False,cbar=False,cmap='viridis')

![image](https://github.com/user-attachments/assets/7223e7cb-9845-467c-8dc8-b116fe82cd0a)

titan.drop('Cabin',axis=1,inplace=True)

titan.head()

![image](https://github.com/user-attachments/assets/ce6683e7-d058-4e1d-9fe2-caadabc33cb1)

titan.dropna(inplace=True)

titan.info()

![image](https://github.com/user-attachments/assets/09caef8e-f3c3-4799-a1aa-10b60948ff1a)

pd.get_dummies(titan['Embarked'],drop_first=True).head()

![image](https://github.com/user-attachments/assets/ce2eae25-3ede-4a80-bcf7-84646cbcdcb5)

sex=pd.get_dummies(titan['Sex'],drop_first=True)
embark=pd.get_dummies(titan['Embarked'],drop_first=True)

titan.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

titan.head()

![image](https://github.com/user-attachments/assets/36c9344a-e30b-4606-a214-ac37250e47a1)

titan=pd.concat([titan,sex,embark],axis=1)

titan.head()

![image](https://github.com/user-attachments/assets/b1a51333-de58-445d-a159-da8136654045)

titan.drop('Survived',axis=1).head()

![image](https://github.com/user-attachments/assets/817860b9-1c6a-418b-955e-1d2e06b48bb7)

titan['Survived'].head()

![image](https://github.com/user-attachments/assets/ee6f7754-6b6f-4475-aff1-c6d4c3f313a4)

from sklearn.model_selection import train_test_split

X_titan,X_test,Y_titan,Y_test = train_test_split(titan.drop('Survived',axis=1),titan['Survived'],test_size=0.30,random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()
logmodel.fit(X_titan,Y_titan)

predictions = logmodel.predict(X_test)

from sklearn.metrics import confusion_matrix

accuracy=confusion_matrix(Y_test,predictions)

accuracy

![image](https://github.com/user-attachments/assets/2d746514-12f6-489c-baf0-85450aa5f6b4)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(Y_test,predictions)
accuracy

predictions

![image](https://github.com/user-attachments/assets/22329052-d469-49ce-9cd2-6a43766b0d96)

# RESULT
        Data analysis was completed successfully
