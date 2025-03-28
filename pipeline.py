from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def create_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # ML model
    ])
    return pipeline
