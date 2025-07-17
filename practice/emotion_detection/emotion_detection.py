import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

#load
df= pd.read_csv('emotion_detection_iris.csv')

x= df.drop('emotion', axis=1)
y = df['emotion']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model= RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

print("confusion matrix:\n", confusion_matrix(y_test, model.predict(x_test)))
print("classification report:\n", classification_report(y_test, model.predict(x_test)))

joblib.dump(model, 'emotion_detection_model.pkl')