import pandas as pd 
import numpy as np

# Seed set kiya taaki har baar run karne par same values generate ho
np.random.seed(42)
n_samples = 2000

# 1. Quantity column ke liye data (1 se 100 ton ke beech)
quantity = np.random.uniform(1.0, 100.0, n_samples) 

# 2. Industrial Rules ke mutabik proportional weights generate karein
# Hum ranges ke hisab se random values generate karenge
ash_raw = np.random.uniform(5.0, 35.0, n_samples)
moisture_raw = np.random.uniform(2.0, 15.0, n_samples)
volatile_raw = np.random.uniform(15.0, 38.0, n_samples)
# Fixed Carbon thermal coal mein aamtaur par 30% se 65% ke beech hota hai
carbon_raw = np.random.uniform(30.0, 65.0, n_samples)

# Sabka total nikalenge taaki normalise kar sakein
total_raw = ash_raw + moisture_raw + volatile_raw + carbon_raw

# 3. Mass Balance Rule: Har component ko strictly 100% ke scale par fit karna
ash_content = (ash_raw / total_raw) * 100
moisture_content = (moisture_raw / total_raw) * 100
volatile_matter = (volatile_raw / total_raw) * 100

# Rounding errors se bachne ke liye Carbon ko bache hue hisse se nikalenge
ash_round = np.round(ash_content, 2)
moisture_round = np.round(moisture_content, 2)
volatile_round = np.round(volatile_matter, 2)

# Yeh ensures karega ki (Carbon + Volatile + Ash + Moisture) ka sum EXACT 100.00 hi aaye
carbon_round = np.round(100.0 - (ash_round + moisture_round + volatile_round), 2)

# Sulphur trace independent component hota hai (0.1% - 4.5%)
sulphur_content = np.random.uniform(0.1, 4.5, n_samples)

# DataFrame creation
coal_df = pd.DataFrame({
    'Quantity (Tonn)': np.round(quantity, 2),
    'Carbon_content (%)': carbon_round,
    'Volatile_Matter (%)': volatile_round,
    'Ash_Content (%)': ash_round,
    'Moisture_content (%)': moisture_round,
    'Sulphur_content (%)': np.round(sulphur_content, 2)
})

# Index=False lagaya taaki CSV mein faltu 'Unnamed: 0' column na bane
coal_df.to_csv("Grade_Data.csv", index=False)

print("Dataset successfully created with 2000 rows & STRICT Mass Balance Check!")
print("\nFirst 5 Rows Preview:")
print(coal_df.head())

# Validation: Check karne ke liye ki total 100% aa raha hai ya nahi
total_percentage = coal_df['Carbon_content (%)'] + coal_df['Volatile_Matter (%)'] + coal_df['Ash_Content (%)'] + coal_df['Moisture_content (%)']
print("\nKya sabhi rows ka total exact 100% hai?:", np.allclose(total_percentage, 100.0))