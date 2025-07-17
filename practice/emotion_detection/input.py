import joblib
import numpy as np

model= joblib.load("emotion_detection_model.pkl")

user_input = input("Enter 4 feature values separated by commas (e.g. 5.1,3.5,1.4,0.2): ")
values = list(map(float, user_input.split(",")))

# Convert to NumPy array for prediction
sample_input = np.array([values])

prediction = model.predict(sample_input)
print("Predicted emotion:", prediction[0])