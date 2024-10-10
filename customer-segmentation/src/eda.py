import matplotlib.pyplot as plt

def exploratory_data_analysis(data):
    # Scatter plot of Annual Income vs Spending Score
    plt.scatter(data['Annual_Income'], data['Spending_Score'])
    plt.title("Annual Income vs Spending Score")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.show()
