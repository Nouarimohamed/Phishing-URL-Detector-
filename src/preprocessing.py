import pandas as pd 
import numpy as np 
from pathlib import Path

# 1. Load dataset
csv_path = Path(__file__).parent / "../data/PhiUSIIL_Phishing_URL_Dataset.csv"
df = pd.read_csv(csv_path)

# 2. Remove duplicate rows
df.drop_duplicates(inplace=True)

# 3. Clean column names
df.columns = df.columns.str.strip().str.lower()

# 4. Handle missing values in numerical columns
numerical_cols = df.select_dtypes(include=np.number).columns

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# 5. Handle missing values in categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    if df[col].notna().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# 6. Remove leading and trailing spaces
for col in categorical_cols:
    df[col] = df[col].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )

# 7. Final verification
print("Missing values:")
print(df.isnull().sum())

print("Duplicates:", df.duplicated().sum())

print("Dataset shape:", df.shape)

# 8. Save cleaned dataset
df.to_csv('cleaned_dataset.csv', index=False)
