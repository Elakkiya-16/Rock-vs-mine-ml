# Rock vs Mine Prediction using Machine Learning

## Overview
This project uses SONAR signal measurements to classify an underwater object as a **Rock (R)** or a **Mine (M)**. It is a supervised binary-classification project.

## Problem Statement
Given 60 numerical SONAR features for an object, predict whether the object is a rock or a mine.

## Dataset
- File: `sonar data.csv`
- Records: 208
- Columns: 61
- Input features: 60 SONAR measurements (columns 0–59)
- Target: column 60 (`R` or `M`)
- The notebook loads the CSV without a header because the first row contains data.

## Technologies and Libraries
- Python
- NumPy
- Pandas
- scikit-learn
- Logistic Regression

## Workflow
1. Import NumPy, Pandas and the required scikit-learn functions.
2. Load the SONAR CSV using `pd.read_csv(..., header=None)`.
3. Inspect the data using `head()`, `shape()`, `describe()` and `value_counts()`.
4. Compare average feature values by class using `groupby(60).mean()`.
5. Separate features (`X`, columns 0–59) and target (`Y`, column 60).
6. Split the data using `train_test_split(test_size=0.1, stratify=Y, random_state=1)`.
7. Create and train `LogisticRegression`.
8. Evaluate training and testing accuracy.
9. Reshape a new 60-value input and use the trained model to predict its class.

## Model
The notebook uses **Logistic Regression** for binary classification.

## Results
| Evaluation | Accuracy |
|---|---:|
| Training accuracy | 83.42% |
| Testing accuracy | 76.19% |

Accuracy is the proportion of predictions that match the actual labels for the evaluated split. Results may vary if the data or split settings change.

## Example Prediction
The notebook supplies one example containing 60 SONAR measurements. The model predicts the class and prints either:
- `The object is a Rock`, or
- `The object is a mine`

The prediction is based on the supplied example input; it is not a guarantee of real-world detection performance.

## How to Run
1. Place `Rock_vs_Mine_Prediction.ipynb` and `sonar data.csv` in an accessible working location.
2. Install the dependencies:
   ```bash
   pip install numpy pandas scikit-learn jupyter
   ```
3. Update the CSV path in the notebook if necessary.
4. Open the notebook in Jupyter Notebook or JupyterLab.
5. Run the cells from top to bottom.

## Notes
- The notebook uses `header=None` because the dataset has no header row.
- The model expects 60 input features in the same order as the dataset.
- The notebook evaluates performance using accuracy; additional metrics such as precision, recall, F1-score and a confusion matrix can provide a fuller evaluation.
