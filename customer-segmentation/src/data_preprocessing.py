import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    # Load dataset
    data = pd.read_csv(filepath)
    
    # Data Cleaning (fill missing values if any)
    data.fillna(method='ffill', inplace=True)
    
    # Standardize the numerical data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Annual_Income', 'Spending_Score']])
    
    return data, scaled_data
