import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('Crop_recommendation.csv')

#split the features and labels
X = df.iloc[:, :-1]  #features
Y = df.iloc[:, -1]  #labels

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier()

model.fit(X_train, Y_train)

predictions = model.predict(X_test)

accuracy = model.score(X_test, Y_test)

print("accuracy:", accuracy)

new_features = [[36, 58, 25, 28.6602, 59.31891, 8.399136, 36.9263]]

predicted_crop = model.predict(new_features)
print('Predicted Crop:', predicted_crop)

# Train your model (as you have done previously)
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Save the trained model to a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as model.pkl")

# Load the model from the pickle file
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Use the loaded model to make predictions
predicted_crop = loaded_model.predict(new_features)
print('Predicted Crop:', predicted_crop)
