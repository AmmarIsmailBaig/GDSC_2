import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

file_path = '.vscode/Social_Network_Ads.csv'  
df = pd.read_csv(file_path)

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print('\nClassification Report:')
print(classification_report(y_test, y_pred))

tree.export_graphviz(model, out_file='vizualier.dot',
                     feature_names=['Age', 'EstimatedSalary'],
                     class_names=['0','1'],
                     label='all',
                     rounded = True,
                     filled = True)

def predict_purchase(age, salary):
    input_data = pd.DataFrame({'Age': [age], 'EstimatedSalary': [salary]})
    prediction = model.predict(input_data)
    return "yes" if prediction[0] == 1 else "No"

user_age = float(input("Enter age: "))
user_salary = float(input("Enter estimated salary: "))

prediction_result = predict_purchase(user_age, user_salary)
print(f"The user is likely to purchase: {prediction_result}")
