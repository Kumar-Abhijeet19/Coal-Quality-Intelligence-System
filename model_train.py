import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Aapki CSV file ka path
file = r"C:\Users\KIIT\Desktop\New folder (5)\Grade_Data.csv"

try:
    df = pd.read_csv(file)
    print("Data Loaded Successfully. Shape:", df.shape)
    print("Available Columns in CSV:", list(df.columns))
except FileNotFoundError:
    print("Error: CSV not found, creating baseline structure.")
    np.random.seed(42)
    n_samples = 2000
    quantity = np.random.uniform(1.0, 100.0, n_samples)
    ash = np.random.uniform(5, 35, n_samples)
    moisture = np.random.uniform(2, 15, n_samples)
    volatile = np.random.uniform(15, 38, n_samples)
    carbon = 100.0 - (ash + moisture + volatile)
    sulphur = np.random.uniform(0.1, 4.5, n_samples)
    
    df = pd.DataFrame({
        'Quantity (Tonn)': np.round(quantity, 2),
        'Carbon_content (%)': np.round(carbon, 2),
        'Volatile_Matter (%)': np.round(volatile, 2),
        'Ash_Content (%)': np.round(ash, 2),
        'Moisture_content (%)': np.round(moisture, 2),
        'Sulphur_content (%)': np.round(sulphur, 2)
    })

# --- DYNAMIC COLUMN MAPPING (Yeh check karega ki CSV mein kaunse naam hain) ---
# Agar CSV mein (%) waale naam hain toh wo use honge, nahi toh purane bina (%) waale use honge
carbon_col = 'Carbon_content (%)' if 'Carbon_content (%)' in df.columns else 'Carbon_content'
volatile_col = 'Volatile_Matter (%)' if 'Volatile_Matter (%)' in df.columns else 'Volatile_Matter'
ash_col = 'Ash_Content (%)' if 'Ash_Content (%)' in df.columns else 'Ash_Content'
moisture_col = 'Moisture_content (%)' if 'Moisture_content (%)' in df.columns else 'Moisture_content'
sulphur_col = 'Sulphur_content (%)' if 'Sulphur_content (%)' in df.columns else 'Sulphur_content'

print(f"\nUsing features: {carbon_col}, {volatile_col}, {ash_col}, {moisture_col}, {sulphur_col}")

# Volatile matter check karne ka logic dynamic columns ke sath
if volatile_col not in df.columns:
    df[volatile_col] = 100 - (df[carbon_col] + df[ash_col] + df[moisture_col])
    df[volatile_col] = df[volatile_col].clip(15.0, 42.0)

# Target (GCV) Generation formula dynamic columns ke sath
df['Actual_GCV'] = (df[carbon_col] * 82) + (df[volatile_col] * 35) - (df[ash_col] * 45) - (df[moisture_col] * 25)
df['Actual_GCV'] += np.random.normal(0, 50, len(df))
df['Actual_GCV'] = df['Actual_GCV'].clip(2000, 7500)

# Features (X) selection bina 'Quantity' ke
X = df[[carbon_col, volatile_col, ash_col, moisture_col, sulphur_col]]
y_gcv = df['Actual_GCV']

# Data splitting (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y_gcv, test_size=0.2, random_state=42)

# Training purely the model
gcv_model = RandomForestRegressor(n_estimators=150, random_state=42)
gcv_model.fit(X_train, y_train)

# Model Evaluation Score
train_score = gcv_model.score(X_train, y_train)
test_score = gcv_model.score(X_test, y_test)
print(f"\nTraining R2 Score: {train_score:.4f}")
print(f"Testing R2 Score: {test_score:.4f}")

# Model save karein
joblib.dump(gcv_model, "coal_grade_model.pkl")
print("\n[SUCCESS] Clean GCV ML Model Saved Successfully!")