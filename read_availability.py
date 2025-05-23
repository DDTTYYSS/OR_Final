import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('Available_boolean.csv')

# Define department mapping
dept_mapping = {
    'AC': 0,
    'GPDA': 1,
    'PR': 2,
    'DM': 3
}

# Get unique values for each dimension
ids = sorted(df['ID'].unique())
depts = sorted(df['dept'].unique())
k_values = sorted(df['k'].unique())
s_values = sorted(df['s'].unique())

# Create a 4D array initialized with zeros
# Shape: (num_ids, num_depts, num_k, num_s)
availability = np.zeros((len(ids), len(depts), len(k_values), len(s_values)), dtype=int)

# Create mapping dictionaries for faster lookup
dept_to_idx = {dept: idx for idx, dept in enumerate(depts)}
k_to_idx = {k: idx for idx, k in enumerate(k_values)}
s_to_idx = {s: idx for idx, s in enumerate(s_values)}

# Fill the array with values from the CSV
for _, row in df.iterrows():
    id_idx = row['ID'] - 1  # Assuming IDs start from 1
    dept_idx = dept_to_idx[row['dept']]
    k_idx = k_to_idx[row['k']]
    s_idx = s_to_idx[row['s']]
    availability[id_idx, dept_idx, k_idx, s_idx] = row['available']


#### 拿去用的時候可以直接註解掉這個 print
print(availability[0, 2, 3, 14])
print(availability[0, 2, 3, 15])
print(availability[0, 2, 3, 16])
print(availability[0, 2, 3, 17])
print(availability[0, 2, 3, 18])
print(availability[0, 2, 3, 19])
print(availability[0, 2, 3, 20])
print(availability[0, 2, 3, 21])
print(availability[0, 2, 3, 22])
