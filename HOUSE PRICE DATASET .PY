import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

# 1. Generate synthetic data using make_regression
#  - n_samples: Number of data points (houses)
#  - n_features: Number of features influencing price (e.g., area, bedrooms, location score)
#  - n_informative: Number of features actually correlated with the target
#  - random_state: For reproducibility
#  - noise: Introduces randomness (how much features affect price)
X, y, = make_regression(
    n_samples=1000, 
    n_features=5, 
    n_informative=3, 
    random_state=42, 
    noise=20
)

# 2. Convert to a Pandas DataFrame for easier manipulation
# Create feature names
feature_names = ['Area', 'Bedrooms', 'Bathrooms', 'Location_Score', 'Year_Built']
df_features = pd.DataFrame(X, columns=feature_names)

# Add target variable (house prices)
df_features['Price'] = y

# 3. Introduce realistic house price range and relationships
# Adjust prices to a more realistic range (e.g., $100k - $1M)
# and add some non-linear dependencies or categorical features if desired
df_features['Price'] = df_features['Price'].apply(lambda x: np.exp(x/1000) * 100000 + 100000) 

# Introduce a categorical feature: 'Neighborhood_Type'
neighborhood_types = ['Urban', 'Suburban', 'Rural']
df_features['Neighborhood_Type'] = np.random.choice(neighborhood_types, size=len(df_features))

# 4. Display the first few rows of the dataset
print(df_features.head())

# 5. Save the dataset to a CSV file (for future use or sharing)
df_features.to_csv('house_price_dataset.csv', index=False) 

print("\nDataset created and saved to 'house_price_dataset.csv'")

