import pandas as pd
from sklearn.model_selection import train_test_split
from svm import Svm
from Logistic_Regression import logistic_regression
from D_tree import de_terr
from sklearn.preprocessing import StandardScaler
import numpy
from sklearn.feature_selection import SelectKBest, chi2
from Knn import knn

data = pd.read_csv('loan_data.csv')
data.drop_duplicates(inplace=True)

# Replacing All NaN Values in Loan Amount with the Mean of it
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mean())

# Replacing All NaN Values in Credit History with the Median of it
data['Credit_History'] = data['Credit_History'].fillna(
    data['Credit_History'].median())
# Drop other NaN Values From all Columns
data.dropna(inplace=True)
# Converting Categorial Data to Numerical

data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
data['Dependents'] = data['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})

data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})

data['Property_Area'] = data['Property_Area'].map(
    {'Urban': 0, 'Rural': 1, 'Semiurban': 2})

data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})

data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})

X = data.iloc[:, 1: 12].values
Y = data.iloc[:, -1].values
# feature selection
kbest = SelectKBest(chi2, k=9)
kbest.fit_transform(X, Y)
print(kbest.get_support())

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=.3, random_state=20)
#data scaling
sc = StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
algo = logistic_regression(X_train, X_test, Y_train, Y_test)
algo.Accuracy()
algo2 = de_terr(X_train, X_test, Y_train, Y_test)
algo2.Accuracy()
algo3 = knn(X_train, X_test, Y_train, Y_test)
algo3.Accuracy()
algo4= Svm(X_train, X_test, Y_train, Y_test)
algo4.Accuracy()