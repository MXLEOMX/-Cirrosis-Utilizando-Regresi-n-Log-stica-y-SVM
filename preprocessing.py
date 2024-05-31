# preprocessing.py
import pandas as pd
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    # Imputar datos faltantes para variables numéricas
    numerical_cols = ['Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']
    imputer_num = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

    # Imputar datos faltantes para variables categóricas
    categorical_cols = ['Ascites', 'Hepatomegaly', 'Spiders', 'Stage']
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

    # Verificar si hay valores NaN después de la imputación
    print("Valores NaN después de la imputación:\n", df[numerical_cols + categorical_cols].isna().sum())

    # Codificar variables categóricas
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    df['Ascites'] = df['Ascites'].map({'Y': 1, 'N': 0})
    df['Hepatomegaly'] = df['Hepatomegaly'].map({'Y': 1, 'N': 0})
    df['Spiders'] = df['Spiders'].map({'Y': 1, 'N': 0})
    df['Edema'] = df['Edema'].map({'Y': 2, 'S': 1, 'N': 0})
    df = pd.get_dummies(df, columns=['Drug'], drop_first=True)
    df['Status'] = df['Status'].map({'C': 0, 'CL': 1, 'D': 2})

    # Separar características y variable objetivo
    X = df.drop(columns=['ID', 'Status'])
    y = df['Status']
    
    # Verificar si hay valores NaN en X
    print("Valores NaN en X después de la codificación:\n", X.isna().sum())
    
    return X, y
