import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('orange_trees.csv')

# Split the data into features (X) and target (y)
X = df.drop(['tree_type', 'price_per_orange'], axis=1)
y, class_names = pd.factorize(df['tree_type'])

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Determine the best max_depth
max_info = {"max_depth": 0, "accuracy": 0}
previous_accuracy = 0

# Iterate through different max_depth values to find the best one
for i in range(1, 20):
    model = DecisionTreeClassifier(max_depth=i, random_state=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Print accuracy for each iteration
    print(f"Iteration {i}: Max Depth = {i}, Accuracy = {accuracy:.2f}")

    # Update max_info if current accuracy is better
    if accuracy > max_info["accuracy"]:
        max_info.update({"max_depth": i, "accuracy": accuracy})

    previous_accuracy = accuracy

# Print the best max_depth and its corresponding accuracy
print(
    f'Best max_depth: {max_info["max_depth"]}, Accuracy: {max_info["accuracy"]}')

# Assign the best max depth to the model
model = DecisionTreeClassifier(max_depth=max_info["max_depth"], random_state=0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Compute and plot the confusion matrix using seaborn
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Read the new data from the CSV file 'unknown_trees.csv'
new_df = pd.read_csv('unknown_trees.csv')

# Extract the same columns from the new data as used during training
# This ensures consistency in the feature columns between training and prediction
X_new = new_df[X.columns]

# Use the trained model to predict the tree types for the new data
predicted_indices = model.predict(X_new)

# Create a new column in the DataFrame to store the predicted tree types
# Map the predicted indices to their corresponding class names using 'class_names'
new_df['predicted_tree_type'] = [class_names[index]
                                 for index in predicted_indices]


# Print the new DataFrame with predicted tree types
print(new_df)
