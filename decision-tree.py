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
y = df['tree_type']

# Convert categorical variable to numerical and keep track of class names
y, class_names = pd.factorize(df['tree_type'])

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Calculate the best max_depth
max_info = {
    "max_depth": 0,
    "accuracy": 0
}

previous_accuracy = 0

for i in range(1, 20):
    model = DecisionTreeClassifier(max_depth=i, random_state=0)
    model.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, predictions)

    if accuracy <= previous_accuracy:
        break  # Early stopping if accuracy doesn't improve

    # Update max_info if current accuracy is better
    if accuracy > max_info["accuracy"]:
        max_info["max_depth"] = i
        max_info["accuracy"] = accuracy

    previous_accuracy = accuracy

print(
    f'Best max_depth: {max_info["max_depth"]}, Accuracy: {max_info["accuracy"]}')


# Compute and plot the confusion matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# Predict values

# Load the new CSV file
new_df = pd.read_csv('unknown_trees.csv')

# Get the features from the new data
X_new = new_df[['tree_height', 'leaf_width', 'root_depth', 'yearly_amount']]

# Make predictions on the new data using the trained model
predicted_indices = model.predict(X_new)

# Convert numerical indices back to original class labels
predicted_tree_types = [class_names[index] for index in predicted_indices]

# Add predictions back to the new dataframe for easy visualization
new_df['predicted_tree_type'] = predicted_tree_types

print(new_df)
