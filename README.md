# Orange Trees Classifier

This project contains a simple Decision Tree classifier to predict the type of orange trees based on certain features.

## Overview

The code loads a dataset from an `orange_trees.csv` file and uses a Decision Tree Classifier to predict the type of orange trees. The classifier iteratively determines the best `max_depth` for the Decision Tree based on the accuracy. After training the model, it computes and displays a confusion matrix to help in assessing the model's performance.

Furthermore, the classifier also predicts tree types for a new dataset from an `unknown_trees.csv` file.

## Dependencies

- Python
- pandas
- scikit-learn
- matplotlib
- seaborn

## Usage

1. Ensure you have the required dependencies installed.
2. Place your dataset in the root directory with the filename `orange_trees.csv`.
3. If you wish to predict new data, make sure the `unknown_trees.csv` is also in the root directory.
4. Run the script to train the model, visualize the confusion matrix, and predict new data.

## Output

Upon running the script:

1. It will display the best `max_depth` used for the Decision Tree and its associated accuracy.
2. A heatmap representing the confusion matrix will be displayed.
3. Predictions for the new dataset (`unknown_trees.csv`) will be printed with a new column `predicted_tree_type` appended to the dataset.
