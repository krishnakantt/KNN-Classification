#importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris # Load the iris dataset
iris = load_iris()

print("Iris dataset loaded successfully!")
print("Features:", iris.feature_names)
print("Target names:", iris.target_names)

# Create a DataFrame with the iris dataset features
df = pd.DataFrame(data=iris.data, columns=iris.feature_names) 
df['target'] = iris.target  # Add the target variable to the DataFrame
print("DataFrame created with iris dataset features:")
print(df.head()) # Display the first few rows of the DataFrame
print(df.shape) # Display the first few rows and shape of the DataFrame
df[df.target == 1].head() # Display the first few rows where target is 0
print("DataFrame shape:", df.shape) # Display the shape of the DataFrame
print("First few rows of the DataFrame where target is 1:")
print(df[df.target == 1].head()) # Display the first few rows where target is 1
print("First few rows of the DataFrame where target is 2:")
print(df[df.target == 2].head()) # Display the first few rows where target is 2
print("First few rows of the DataFrame where target is 0:")
print(df[df.target == 0].head()) # Display the first few rows where target is 0
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x]) # Add a new column with flower names
print("New column 'flower_name' added to the DataFrame:")

df0=df[:50] # Select the first 50 rows
df1=df[50:100] # Select the next 50 rows
df2=df[100:] # Select the last 50 rows

#Scatter plot of Sepal Length vs Sepal Width
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Iris Dataset Sepal Length vs Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='red', label='Setosa')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='blue', label='Versicolor')

#Train and test split
from sklearn.model_selection import train_test_split
X = df.drop(['target','flower_name'], axis='columns')  # Features excluding target and flower name
y = df.target  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("Train and test split completed.")

#Create KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)  # Create KNN model with 3 neighbors
knn.fit(X_train, y_train)  # Fit the model on the training data
print("KNN model created and fitted on training data.")

# Make predictions on the test set
y_pred = knn.predict(X_test)  # Predict the target for the test set
print("Predictions made on the test set.")

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print("Model accuracy:", accuracy)  # Print the accuracy
print("Classification report:\n", classification_report(y_test, y_pred))  # Print classification