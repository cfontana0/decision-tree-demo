# Orange Trees Classifier

This project implements a Decision Tree classifier to predict the type of orange trees based on specific features.

## Overview

The code reads a dataset from the `orange_trees.csv` file and employs a Decision Tree Classifier to predict the orange tree types. The classifier iteratively identifies the optimal `max_depth` for the Decision Tree by assessing accuracy. Once the model is trained, it calculates and displays a confusion matrix to evaluate performance.

Moreover, the classifier predicts tree types for a new dataset using the `unknown_trees.csv` file.

## Dependencies

Make sure you have the following dependencies installed before running the project:

- Python
- pandas
- scikit-learn
- matplotlib
- seaborn

You can easily install these dependencies by executing:

```bash
pip install -r requirements.txt
```

## Usage

Follow these steps to run the project:

1. Install the required dependencies using the command mentioned above.

2. Place your dataset in the root directory with the filename `orange_trees.csv`.

3. If you intend to make predictions for new data, ensure that the `unknown_trees.csv` file is also present in the root directory.

4. Execute the script to train the model, visualize the confusion matrix, and predict new data. You can choose either of the following commands:

```bash
python3 decision-tree.py
```

or

```bash
python decision-tree.py
```

## Output

Upon running the script:

1. The best `max_depth` used for the Decision Tree along with its corresponding accuracy will be displayed.

2. A heatmap illustrating the confusion matrix will be shown.

3. Predictions for the new dataset (`unknown_trees.csv`) will be printed, and a new column labeled `predicted_tree_type` will be appended to the dataset.