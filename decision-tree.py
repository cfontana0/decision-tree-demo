import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('orange_trees.csv')

# Split the data into features (X) and target (y)
X = df.drop(['tree_type', 'price_per_orange'], axis=1)
y = df['tree_type']

# Convert categorical variable to numerical and keep track of class names
y, class_names = pd.factorize(df['tree_type'])

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Calculate the best max_depth
max_info = {
    "max_depth": 0,
    "accuracy": 0
}

for i in range(1, 20):
    model = DecisionTreeClassifier(max_depth=i, random_state=0)
    model.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, predictions)
    if accuracy > max_info["accuracy"]:
        max_info["max_depth"] = i
        max_info["accuracy"] = accuracy

print(f'Max: {max_info}')

# Train the model
model = DecisionTreeClassifier(max_depth=max_info["max_depth"], random_state=0)
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, predictions)


# Compute and plot the confusion matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()