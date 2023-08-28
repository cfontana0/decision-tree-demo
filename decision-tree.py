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

for i in range(1, 20):
    model = DecisionTreeClassifier(max_depth=i, random_state=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Iteration {i}: Max Depth = {i}, Accuracy = {accuracy:.2f}")

    # Update max_info if current accuracy is better
    if accuracy > max_info["accuracy"]:
        max_info.update({"max_depth": i, "accuracy": accuracy})

    previous_accuracy = accuracy

print(
    f'Best max_depth: {max_info["max_depth"]}, Accuracy: {max_info["accuracy"]}')

# Assign the best max depth
model = DecisionTreeClassifier(max_depth=max_info["max_depth"], random_state=0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Compute and plot the confusion matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Predict values for new data
new_df = pd.read_csv('unknown_trees.csv')
X_new = new_df[X.columns]  # Use same columns as before for consistency
predicted_indices = model.predict(X_new)
new_df['predicted_tree_type'] = [class_names[index]
                                 for index in predicted_indices]

print(new_df)
