#!/usr/bin/env python
# coding: utf-8

# In[3]:


import requests
import json

# Specify FRED API endpoint and data series ID
api_endpoint = 'https://api.stlouisfed.org/fred/series/observations'
data_series_id = 'GDP'

# Specify API key (if applicable)
api_key = 'c0804bb7a9d5d7d79922cb053b604f61'

# Parameters for API request
params = {
    'series_id': data_series_id,
    'api_key': api_key,  # Include API key if registered
    'file_type': 'json',  # Specify file type (json, xml)
    'observation_start': '2010-01-01',  # Start date for data retrieval
    'observation_end': '2022-12-31',  # End date for data retrieval
}

# Make API request
response = requests.get(api_endpoint, params=params)

# Parse JSON response
data = response.json()

# Extract relevant data
observations = data['observations']

# Print first few observations
print(json.dumps(observations[:5], indent=4))  # Print first 5 observations


# In[7]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Assuming 'observations' contains the retrieved data
df = pd.DataFrame(observations)

# Data Cleaning
# Remove missing values
df.dropna(inplace=True)

# Convert 'value' column to numeric type
df['value'] = pd.to_numeric(df['value'], errors='coerce')

# Drop rows with missing values after conversion
df.dropna(subset=['value'], inplace=True)

# Handle outliers (e.g., winsorization)
# Define thresholds for winsorization
lower_bound = df['value'].quantile(0.05)
upper_bound = df['value'].quantile(0.95)

# Winsorize outliers
df['value'] = df['value'].clip(lower=lower_bound, upper=upper_bound)

# Preprocessing
# Data Transformation (e.g., log transformation)
df['log_value'] = np.log(df['value'])

# Normalization (e.g., Z-score normalization)
scaler = StandardScaler()
df['scaled_value'] = scaler.fit_transform(df[['value']])

# Feature Engineering (if applicable)
# Example: Extract year and month from date column
df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month

# Processed DataFrame ready for analysis and visualization
df.head()


# In[ ]:




