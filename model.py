import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import pickle

# Load dataset
df = pd.read_csv('water_potability.csv')

# Convert object columns to numeric
numeric_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Split dataset into features (X) and target (y)
X = df.drop('Potability', axis=1)  # Assuming 'Potability' is the target column
y = df['Potability']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train the XGBoost model
model = XGBClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print("Model saved as model.pkl")

