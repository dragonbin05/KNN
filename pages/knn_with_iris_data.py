import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import streamlit as st

st.write("# KNN With :blue[Iris Dataset]")

# Load Iris dataset
iris = load_iris() 
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target
# 0 → Iris-setosa
# 1 → Iris-versicolor
# 2 → Iris-virginica

# Split the dataset into features and target variable
x = data.drop('target', axis=1)
y = data['target']

data['target'] = data['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
st.write(data)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2025)

# Standardize the features (using z-score normalization)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

st.write("train size:", x_train.shape[0])
st.write("test size:", x_test.shape[0])

# Create and train the KNN classifier
k = st.slider("Select number of neighbors (`k`)", min_value=1, max_value=15, value=5)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

# Make predictions
y_pred = knn.predict(x_test)

# Calculate scores
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

st.write(f"### Accuracy: :red[{accuracy:.2f}]")
st.write(f"### Balanced Accuracy: :red[{balanced_accuracy:.2f}]")
st.write(f"### F1 Score: :red[{f1:.2f}]")

# User input for prediction
st.write("---")
st.write("## Predict Iris Species")
input = st.text_input("Enter sepal length, sepal width, petal length, petal width (comma separated):", "5.1,3.5,1.4,0.2")

try:
    input_data = [float(i) for i in input.split(",")]
    input_data = scaler.transform([input_data])
    prediction = knn.predict(input_data)
    st.write(f"#### Predicted Iris Species: `{iris.target_names[prediction][0]}`")

except:
    st.warning("Please enter valid input data.")