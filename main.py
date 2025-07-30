# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Try Block to improve error handling
try:
    # Using pandas to read the csv file and storing it in df variable
    df = pd.read_csv('Kaggle_Telco_Customer_Churn_DataSet.csv')

    # Doing Initial Inspection/Exploration
    print(
        df.info(),
        df.head(),
        df.tail(),
    )

    # Used Below to check if there's some data types other than float or whole column is mistakenly tagged as object dtype
    # df_total_charges_object_element_list = [x for x in df['TotalCharges'] if type(x) != float]
    # print(len(df_total_charges_object_element_list))
    # Whole Column was marked as object but every value seems float

    # Data Cleaning in Process - Changing the empty strings with NaN then changing the dtype to float and at last replacing NaN with mean of the column
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df['TotalCharges'] = df['TotalCharges'].replace(np.nan, np.mean(df['TotalCharges']))

    # Splitting Data into features and target
    x = df.drop(['customerID','Churn'],axis=1)
    y = df['Churn']

    # One Hot Encoding for required features and target
    x = pd.get_dummies(x,
                       columns = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod'],
                       prefix = ['gender_', 'Partner_', 'Dependents_', 'PhoneService_', 'MultipleLines_', 'InternetService_', 'OnlineSecurity_', 'OnlineBackup_', 'DeviceProtection_',  'TechSupport_', 'StreamingTV_', 'StreamingMovies_', 'Contract_',  'PaperlessBilling_', 'PaymentMethod_']
                       )
    y = y.replace({'Yes':1,'No':0})

    # Splitting data into train/test
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    # Scaling x_train
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Storing the Logistic Regression model inside model variable and fitting it to the training dataset
    model = LogisticRegression()
    model.fit(x_train,y_train)

    # Predicting
    y_pred = model.predict(x_test)

    # Evaluating Performance - confusion_matrix, f1_score, precision_score, recall_score
    cnfson_mtrx = confusion_matrix(y_test, y_pred)
    print(
        f'confusion_matrix - \n{cnfson_mtrx}\n'
        f', f1_score - {f1_score(y_test,y_pred)}\n'
        f', precision_score - {precision_score(y_test,y_pred)}\n'
        f', recall_score - {recall_score(y_test,y_pred)}\n'
    )

    # Visualising using a heatmap
    plt.figure(figsize=(10,10))
    sns.heatmap(
        cnfson_mtrx,
        annot = True,
        cmap='YlGnBu',
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix')
    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.show()

except Exception as e:
    print(f'An Exception occurred : {e}')