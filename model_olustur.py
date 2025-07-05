import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# 1. CSV oku
df = pd.read_csv("House_Rent_Dataset.csv")

# 2. Sayıya çevrilmesi gereken kolonlar
df["Size"] = pd.to_numeric(df["Size"], errors="coerce")
df["Rent"] = pd.to_numeric(df["Rent"], errors="coerce")

# 3. NaN'leri temizle
df.dropna(inplace=True)

# 4. Gereksiz sütunları kaldır
df.drop(columns=["Posted On", "Floor", "Area Locality", "Tenant Preferred", "Point of Contact"], errors="ignore", inplace=True)

# 5. Aykırı büyüklükleri temizle
df = df[(df["Size"] >= 30) & (df["Size"] < 500)]

# 6. Kategorikleri dönüştür
df = pd.get_dummies(df, columns=["Furnishing Status", "City", "Area Type"], drop_first=True)

# 7. X ve y ayır
X = df.drop("Rent", axis=1)
y = np.log(df["Rent"])

# 8. Eğitim/Test ayır
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Ölçekle
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 10. Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# 11. Kaydet
joblib.dump(model, "random_forest_model.pkl")
joblib.dump(scaler, "veri_olcekleyici.pkl")
pd.Series(X.columns).to_csv("columns.csv", index=False, header=False)
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("✅ Her şey yolunda! Model, scaler ve sütunlar kaydedildi.")


