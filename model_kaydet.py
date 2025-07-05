import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Dataset'i oku
df = pd.read_csv("House_Rent_Dataset.csv")

# Kategorik değişkenleri one-hot encode et
df_encoded = pd.get_dummies(df, columns=['Furnishing Status', 'City', 'Area Type'], drop_first=True)

# Özellik ve hedef değişken
X = df_encoded.drop("Rent", axis=1)
y = np.log(df_encoded["Rent"])  # Log dönüşümü

# Veriyi ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğit
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Kaydet
joblib.dump(model, "random_forest_model.pkl")
print("✅ Model başarıyla kaydedildi: random_forest_model.pkl")
