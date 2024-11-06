import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load dataset and prepare the model as before
data = pd.read_csv('diabetes.csv')
X = data[['Glucose', 'BMI', 'Insulin']]
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
