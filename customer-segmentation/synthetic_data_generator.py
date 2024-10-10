import pandas as pd
import numpy as np

# Setting random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_customers = 200

# CustomerID
customer_id = np.arange(1, num_customers + 1)

# Gender (Randomly assign Male/Female)
gender = np.random.choice(['Male', 'Female'], size=num_customers)

# Age (Random ages between 18 and 70)
age = np.random.randint(18, 70, size=num_customers)

# Annual Income (Random income between 15K and 150K)
annual_income = np.random.randint(15000, 150000, size=num_customers)

# Spending Score (Random scores between 1 and 100)
spending_score = np.random.randint(1, 100, size=num_customers)

# Creating a DataFrame
data = pd.DataFrame({
    'CustomerID': customer_id,
    'Gender': gender,
    'Age': age,
    'Annual_Income': annual_income,
    'Spending_Score': spending_score
})

# Save the dataset to a CSV file
data.to_csv('data/synthetic_customer_data.csv', index=False)

print("Synthetic customer dataset created!")
