import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load all files
df1 = pd.read_csv("datatraining.txt")
df2 = pd.read_csv("datatest.txt")
df3 = pd.read_csv("datatest2.txt")

# Merge files
data = pd.concat([df1, df2, df3], ignore_index=True)

X = data[['Temperature','Humidity','Light','CO2','HumidityRatio']]
y = data['Occupancy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

print("Model Trained Successfully")