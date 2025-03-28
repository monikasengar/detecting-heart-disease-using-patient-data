import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from pipeline import create_pipeline

# Load Data
data = pd.read_csv("data/dataset.csv")

# Split Data
X = data.drop(columns=['target'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and Train Pipeline
pipeline = create_pipeline()
pipeline.fit(X_train, y_train)

# Save Model
joblib.dump(pipeline, 'models/heart_disease_pipeline.pkl')

print("Model trained and saved successfully!")
