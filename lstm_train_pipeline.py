#!/usr/bin/env python3
import os
import utils_general_porpose
import pandas as pd
import utils_split_dataset

#Load data
current_dir = os.getcwd()
print("Current directory:", current_dir)
data = utils_general_porpose.load_json(current_dir, "/concepts/clinical_concepts_20250404_171428.json")
print("Data loaded:", data[:2])
print("Data length:", len(data))

df = pd.DataFrame(data)
df.head()
df.info()

X = df[["id_cliente", "entities"]]
y = df[["label"]]

print(X.head())
print(y.head())

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train data shape:", X_train.shape, y_train.shape)
print("Test data shape:", X_test.shape, y_test.shape)
print("Train data:", X_train.head())
print("Test data:", X_test.head())
print("Train labels:", y_train.head())
print("Test labels:", y_test.head())
print("Train labels unique values:", y_train["label"].unique())
print("Test labels unique values:", y_test["label"].unique())
print("Train labels counts:", y_train["label"].value_counts())
print("Test labels counts:", y_test["label"].value_counts())

balanced_patients = utils_split_dataset.balanced_subsample(data)
print(type(balanced_patients))
print("Balanced patients:", balanced_patients[:10])

patient_train, patient_test = utils_split_dataset.split_data(balanced_patients, 0.8)
