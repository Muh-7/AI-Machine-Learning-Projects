# Breast Cancer Survival Classification

## Code Review & Improvement Roadmap

### 1. Project Overview

This project implements a binary classification model using the **Haberman Survival Dataset**.

The current notebook explores the dataset, prepares the features and target, builds a small neural network using TensorFlow/Keras, trains the model, evaluates its accuracy, and visualizes the learning curves.

The project is useful as a learning experiment for understanding the complete machine learning workflow:

```text
Load Dataset
    ↓
Explore Data
    ↓
Prepare Features and Target
    ↓
Encode Target
    ↓
Train/Test Split
    ↓
Build Neural Network
    ↓
Train
    ↓
Evaluate
    ↓
Visualize Learning Curves
```

However, several improvements are required before the project can be considered a robust machine learning experiment.

---

# 2. Dataset

The notebook currently loads:

```python
haberman.csv
```

Dataset shape:

```text
306 samples
4 columns
```

After assigning meaningful column names:

```python
columns = ['Age', 'Year', 'Nodes', 'Class']
```

the dataset consists of three input features and one target variable.

### Features

| Feature | Description                                |
| ------- | ------------------------------------------ |
| `Age`   | Patient age                                |
| `Year`  | Year of operation                          |
| `Nodes` | Number of detected positive axillary nodes |
| `Class` | Survival class                             |

The original target values are:

```text
Class 1: 225 samples → 73.53%
Class 2: 81 samples  → 26.47%
```

This clearly shows that the dataset is **imbalanced**.

---

# 3. Current Data Preparation

The current notebook separates the features and labels using:

```python
X = data_names.values[:, :-1]
Y = data_names.values[:, -1]
```

The input features are converted to:

```python
float32
```

using:

```python
X = X.astype('float32')
```

The target is encoded using:

```python
Y = LabelEncoder().fit_transform(Y)
```

which transforms:

```text
Original class 1 → 0
Original class 2 → 1
```

Therefore, after preprocessing:

```text
0 = majority class
1 = minority class
```

---

# 4. Current Train/Test Split

The notebook currently uses:

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=3,
    stratify=Y
)
```

Using:

```python
stratify=Y
```

is a good decision because it preserves approximately the same class distribution in both training and testing datasets.

---

# 5. Current Neural Network Architecture

The final model in the notebook is:

```python
model = Sequential([
    Input(shape=(n_features,)),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

Architecture:

```text
Input
3 features
    ↓
Dense Layer
10 neurons
ReLU
    ↓
Output Layer
1 neuron
Sigmoid
```

This is an appropriately small neural network considering that the dataset contains only 306 samples.

A much deeper neural network would probably be unnecessary for such a small tabular dataset.

---

# 6. Model Configuration

The current model uses:

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

These choices are technically appropriate for binary classification.

### Optimizer

```text
Adam
```

### Loss

```text
Binary Cross-Entropy
```

### Output Activation

```text
Sigmoid
```

---

# 7. Current Training Configuration

The model is trained using:

```python
history = model.fit(
    X_train,
    Y_train,
    epochs=200,
    batch_size=20,
    validation_split=0.2,
    validation_data=(X_test, Y_test),
    verbose=1
)
```

Current configuration:

```text
Epochs     = 200
Batch size = 20
Optimizer  = Adam
```

Near the end of training, the notebook recorded approximately:

```text
Training Accuracy   ≈ 75–76%
Validation Accuracy ≈ 80.65%

Training Loss       ≈ 0.52
Validation Loss     ≈ 0.46–0.47
```

---

# 8. Current Evaluation

The current notebook calculates predictions using:

```python
Y_pred = (model.predict(X_test) > 0.5).astype(int)
```

and evaluates:

```python
accuracy_score(Y_test, Y_pred)
```

The model achieved approximately:

```text
Accuracy ≈ 0.81
```

which corresponds to:

```text
Accuracy ≈ 81%
```

---

# 9. Important Bug: Incorrect Accuracy Formatting

The notebook currently contains:

```python
print(f"Accuracy: {score:.2f}%")
```

which produces:

```text
Accuracy: 0.81%
```

This is incorrect because `accuracy_score()` returns a value between `0` and `1`.

The actual result is approximately:

```text
81%
```

### Fix

Use either:

```python
print(f"Accuracy: {score:.2%}")
```

or:

```python
print(f"Accuracy: {score * 100:.2f}%")
```

Expected output:

```text
Accuracy: 80.65%
```

---

# 10. Major Issue: Feature Scaling Is Missing

The three features have significantly different numerical ranges.

For example:

```text
Age   → approximately 30–83
Year  → approximately 58–69
Nodes → approximately 0–52
```

The current neural network receives these values without normalization or standardization.

This can make optimization more difficult and may partially explain the very high loss during the first training epochs.

For example, the initial training loss was approximately:

```text
14.17
```

### Recommended Solution

Use:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Important:

```python
scaler.fit(X_train)
```

should be performed only on the training data.

The test set must not be used to calculate scaling statistics.

---

# 11. Major Issue: Test Set Used as Validation Data

Currently:

```python
validation_data=(X_test, Y_test)
```

is passed during model training.

This means the test dataset is repeatedly evaluated during training.

Even though validation data is not directly used for gradient updates, this compromises the role of the test set as completely unseen final evaluation data, especially when model decisions are made after observing its performance.

A better structure is:

```text
Training Set
    ↓
Validation Set
    ↓
Model Development
    ↓
Final Test Set
```

For example:

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

X_train, X_val, Y_train, Y_val = train_test_split(
    X_train,
    Y_train,
    test_size=0.2,
    random_state=42,
    stratify=Y_train
)
```

Then:

```python
model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val)
)
```

The test set should only be used after model development has finished.

---

# 12. Redundant Validation Configuration

The notebook currently uses both:

```python
validation_split=0.2
```

and:

```python
validation_data=(X_test, Y_test)
```

in the same `model.fit()` call.

When explicit validation data is supplied, `validation_split` is unnecessary.

Only one validation strategy should be used.

Recommended:

```python
model.fit(
    X_train,
    Y_train,
    epochs=200,
    batch_size=20,
    validation_data=(X_val, Y_val)
)
```

---

# 13. Duplicate Model Definition

The notebook currently defines the model twice.

First:

```python
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(n_features,)))
model.add(Dense(1, activation='sigmoid'))
```

Then:

```python
model = Sequential([
    Input(shape=(n_features,)),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

The second implementation is cleaner and follows current Keras recommendations.

The first model definition should therefore be removed.

The final code should simply use:

```python
model = Sequential([
    Input(shape=(n_features,)),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

---

# 14. Class Imbalance

The dataset distribution is:

```text
Class 0 ≈ 73.5%
Class 1 ≈ 26.5%
```

Therefore, accuracy alone can be misleading.

A model predicting the majority class for every sample could already achieve approximately:

```text
74% accuracy
```

without actually learning useful minority-class classification.

This makes additional classification metrics necessary.

---

# 15. Additional Evaluation Results

A later evaluation of the model produced:

```text
              precision    recall  f1-score   support

0                 0.80      0.98      0.88        46
1                 0.83      0.31      0.45        16

accuracy                              0.81        62
macro avg         0.82      0.65      0.67        62
weighted avg      0.81      0.81      0.77        62
```

Confusion matrix:

```text
[[45  1]
 [11  5]]
```

These metrics were calculated after the original notebook version and should be added to a future revision.

---

# 16. Interpretation of the Current Model

The confusion matrix means:

```text
45 majority-class samples correctly classified
1 majority-class sample incorrectly classified

5 minority-class samples correctly classified
11 minority-class samples incorrectly classified
```

The most important observation is:

```text
Class 1 Recall = 0.31
```

The model detects only approximately:

```text
31%
```

of the minority-class samples.

At the same time:

```text
Class 1 Precision = 0.83
```

This means that when the model predicts class `1`, it is usually correct.

However, it predicts class `1` too conservatively and misses many real class-1 samples.

The model therefore currently exhibits approximately:

```text
High minority-class precision
            +
Low minority-class recall
```

This is more informative than simply reporting:

```text
Accuracy = 81%
```

---

# 17. Balanced Accuracy

Using the class recalls:

```text
Class 0 Recall ≈ 0.98
Class 1 Recall ≈ 0.31
```

balanced accuracy is approximately:

```text
(0.98 + 0.31) / 2
≈ 0.645
```

or roughly:

```text
64.5%
```

This shows why the raw 81% accuracy should not be interpreted alone.

---

# 18. Recommended Evaluation Metrics

Future versions should evaluate:

```text
Accuracy
Precision
Recall
F1-score
Confusion Matrix
Balanced Accuracy
ROC-AUC
ROC Curve
Precision-Recall Curve
```

Example:

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score
)
```

---

# 19. Class Weighting Experiment

Because the dataset is imbalanced, class weighting should be investigated.

Using:

```python
from sklearn.utils.class_weight import compute_class_weight
```

the calculated training weights were approximately:

```python
{
    0: 0.6816,
    1: 1.8769
}
```

This means errors on the minority class receive significantly more weight during training.

They can be passed to Keras using:

```python
model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=200,
    batch_size=20,
    class_weight=class_weights
)
```

The objective should not necessarily be to maximize raw accuracy.

A useful improvement would be increasing:

```text
Class 1 Recall
Class 1 F1-score
Balanced Accuracy
```

while maintaining acceptable overall performance.

---

# 20. Decision Threshold Experiment

The current classification threshold is:

```python
0.5
```

because predictions are generated using:

```python
Y_pred = (Y_probability >= 0.5).astype(int)
```

This threshold should not automatically be assumed to be optimal.

Future experiments can evaluate:

```text
0.30
0.35
0.40
0.45
0.50
0.55
...
```

For example:

```python
Y_probability = model.predict(X_test).ravel()

Y_pred = (Y_probability >= 0.4).astype(int)
```

Lowering the threshold generally tends to:

```text
Recall ↑
Precision ↓
```

The threshold should therefore be selected according to the project objective rather than accuracy alone.

Threshold selection should be performed using validation data, not the final test set.

---

# 21. Neural Network May Not Be the Best Model

The dataset contains only:

```text
306 samples
3 input features
```

This is a very small tabular dataset.

A neural network is useful as a learning experiment, but it should not automatically be assumed to be the best machine learning model.

Future versions should compare the neural network against traditional models such as:

```text
Logistic Regression
Support Vector Machine
K-Nearest Neighbors
Decision Tree
Random Forest
```

A strong experiment would compare all models using the same preprocessing and evaluation procedure.

---

# 22. Add a Baseline Model

Before evaluating sophisticated models, create a baseline.

For example:

```python
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy='most_frequent')

baseline.fit(X_train, Y_train)

baseline_pred = baseline.predict(X_test)
```

This establishes how much improvement the actual model provides over a trivial classifier.

Because the majority class represents roughly 74% of the dataset, this comparison is especially important.

---

# 23. Use Cross-Validation

Because the dataset contains only 306 samples, performance may depend significantly on one particular train/test split.

A better evaluation should include:

```text
Stratified K-Fold Cross Validation
```

For example:

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

This would provide a more reliable estimate of model performance than relying on a single split.

---

# 24. Add Early Stopping

Training for exactly:

```text
200 epochs
```

is arbitrary.

Instead, the model should determine when training should stop based on validation performance.

Example:

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)
```

Then:

```python
model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=500,
    batch_size=20,
    callbacks=[early_stopping]
)
```

This reduces unnecessary training and automatically restores the best model parameters.

---

# 25. Improve Reproducibility

Currently:

```python
random_state=3
```

controls the Scikit-learn split, but TensorFlow and NumPy randomness are not explicitly controlled.

A future version should use:

```python
import random
import numpy as np
import tensorflow as tf

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

Then use the same constant throughout:

```python
random_state=SEED
```

This makes experiments easier to reproduce.

---

# 26. Improve Dataset Validation

Before training, future versions should explicitly investigate:

```text
Missing values
Duplicated samples
Invalid values
Feature distributions
Outliers
Class distribution
Feature correlations
```

For example:

```python
data_names.info()
```

```python
data_names.isnull().sum()
```

```python
data_names.duplicated().sum()
```

These checks make the data exploration stage more complete.

---

# 27. Improve Exploratory Data Analysis

The current notebook uses:

```python
data.hist()
```

This is a useful beginning but could be improved.

Recommended visualizations include:

```text
Feature histograms by class
Class distribution
Box plots
Correlation matrix
Scatter plots
Nodes distribution
Age distribution by survival class
```

Plots should also use the meaningful feature names instead of numeric column labels.

---

# 28. Improve Variable Naming

The current code uses:

```python
X
Y
X_train
Y_train
```

These are standard ML names and are acceptable.

However, Python convention normally uses lowercase variable names:

```python
X_train
X_test
y_train
y_test
```

rather than:

```python
Y_train
Y_test
```

Recommended:

```python
X_train, X_test, y_train, y_test
```

---

# 29. Remove Debugging Cells

The notebook currently contains a manual loop such as:

```python
a = 0
b = 0

for i in targets:
    ...
```

and prints the entire target array.

This was useful during learning and debugging but is unnecessary in the final GitHub version because `Counter` already provides the required information.

The final notebook should contain only code that contributes to the analysis.

---

# 30. Improve Comments

Some comments currently state that conversion is required because:

```text
the model will not accept other data types
```

This explanation is too broad.

A better explanation would be:

```python
# Convert input features to float32 for efficient numerical
# computation and compatibility with the TensorFlow model.
X = X.astype("float32")
```

And:

```python
# Convert the original class labels {1, 2}
# into binary labels {0, 1}.
encoder = LabelEncoder()
y = encoder.fit_transform(y)
```

Comments should explain **why** an operation exists rather than repeat exactly what the code does.

---

# 31. Preserve the Label Encoder

Instead of:

```python
Y = LabelEncoder().fit_transform(Y)
```

use:

```python
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
```

This allows future inspection of the mapping:

```python
label_encoder.classes_
```

and prevents the encoding logic from being hidden.

---

# 32. Improve File Loading

The current notebook assumes:

```python
path = r"haberman.csv"
```

exists in the current working directory.

For a GitHub repository, a clearer structure would be:

```text
project/
│
├── data/
│   └── haberman.csv
│
├── notebooks/
│   └── breast_cancer_survival.ipynb
│
├── src/
│
├── requirements.txt
├── README.md
└── MODEL_REVIEW_AND_IMPROVEMENT.md
```

Then dataset paths can be managed more clearly.

---

# 33. Recommended Repository Structure

A future version of the project could use:

```text
breast-cancer-survival-classification/
│
├── README.md
├── MODEL_REVIEW_AND_IMPROVEMENT.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── haberman.csv
│
├── notebooks/
│   └── 01_experiment.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   └── .gitkeep
│
└── results/
    ├── figures/
    └── metrics/
```

The project does not need this entire structure immediately.

The notebook alone is acceptable for the current learning stage.

The structure becomes useful as the experiment grows.

---

# 34. Add `requirements.txt`

The repository should document its dependencies.

A first version may include:

```text
numpy
pandas
matplotlib
scikit-learn
tensorflow
```

For reproducible experiments, dependency versions should eventually be pinned to the versions actually used to execute the project.

---

# 35. Improve GitHub Documentation

The repository README should eventually explain:

```text
Project objective
Dataset
Features
Target meaning
Dataset source
Class distribution
Preprocessing
Models
Evaluation methodology
Results
Limitations
How to run the notebook
Future improvements
```

It is especially important to describe the target correctly.

This project uses the Haberman Survival Dataset and should not simply be described as:

```text
Cancer vs No Cancer
```

It is a **survival classification problem associated with patients who underwent breast cancer surgery**.

A more precise project title would therefore be:

```text
Breast Cancer Survival Classification Using the Haberman Dataset
```

or:

```text
Haberman Survival Classification
```

---

# 36. Current Strengths

The current notebook already demonstrates several good practices:

* Clear progression from data loading to model evaluation.
* Meaningful feature names are introduced.
* Dataset shape and descriptive statistics are inspected.
* Class imbalance is explicitly investigated.
* `stratify` is used during train/test splitting.
* A small neural network is used instead of an unnecessarily deep architecture.
* `sigmoid` is correctly used for the binary output.
* `binary_crossentropy` is correctly used as the loss function.
* Adam is an appropriate initial optimizer.
* Learning curves are visualized.
* Predictions are converted from probabilities to binary classes.
* The project is simple enough to understand and revise later.

---

# 37. Current Weaknesses

The most important weaknesses are:

1. No feature scaling.
2. Test data is used as validation data.
3. `validation_split` and `validation_data` are used together.
4. Accuracy is printed incorrectly.
5. Accuracy is the only metric saved in the current notebook.
6. Minority-class recall is poor.
7. Class imbalance is identified but not handled in the uploaded version.
8. No baseline model.
9. No comparison with classical ML algorithms.
10. No cross-validation.
11. No early stopping.
12. Model architecture is defined twice.
13. Debugging code remains in the notebook.
14. Reproducibility is incomplete.
15. Dataset validation and EDA are limited.
16. No ROC-AUC or Precision-Recall analysis.
17. No final clean separation between training, validation, and testing.
18. Dependency and execution documentation are missing.

---

# 38. Recommended Version 2 Pipeline

The next version should follow approximately:

```text
Load Haberman Dataset
        ↓
Assign Feature Names
        ↓
Data Validation
        ↓
EDA
        ↓
Class Distribution
        ↓
Encode Target
        ↓
Train / Validation / Test Split
        ↓
StandardScaler
        ↓
Baseline Classifier
        ↓
Traditional ML Models
        ↓
Neural Network
        ↓
Class Weight Experiment
        ↓
Early Stopping
        ↓
Threshold Analysis
        ↓
Cross-Validation
        ↓
Final Evaluation
        ↓
Model Comparison
```

---

# 39. Recommended Model Comparison

Future experiments should generate a comparison similar to:

| Model               | Accuracy | Precision | Recall | F1 | ROC-AUC | Balanced Accuracy |
| ------------------- | -------: | --------: | -----: | -: | ------: | ----------------: |
| Dummy Baseline      |        — |         — |      — |  — |       — |                 — |
| Logistic Regression |        — |         — |      — |  — |       — |                 — |
| SVM                 |        — |         — |      — |  — |       — |                 — |
| KNN                 |        — |         — |      — |  — |       — |                 — |
| Random Forest       |        — |         — |      — |  — |       — |                 — |
| Neural Network      |        — |         — |      — |  — |       — |                 — |
| NN + Class Weights  |        — |         — |      — |  — |       — |                 — |

Results should be populated only after running the corresponding experiments.

---

# 40. Priority Roadmap

## Priority 1 — Fix correctness

* [ ] Correct the accuracy formatting.
* [ ] Remove duplicate model creation.
* [ ] Create separate training, validation, and test sets.
* [ ] Remove the test set from `model.fit()`.
* [ ] Remove redundant `validation_split`.
* [ ] Add `StandardScaler`.

## Priority 2 — Improve evaluation

* [ ] Add confusion matrix.
* [ ] Add classification report.
* [ ] Add Precision.
* [ ] Add Recall.
* [ ] Add F1-score.
* [ ] Add Balanced Accuracy.
* [ ] Add ROC-AUC.
* [ ] Plot ROC curve.
* [ ] Plot Precision-Recall curve.

## Priority 3 — Handle imbalance

* [ ] Train the original model.
* [ ] Train with class weights.
* [ ] Compare minority-class recall.
* [ ] Compare F1-score.
* [ ] Experiment with probability thresholds.

## Priority 4 — Improve experimentation

* [ ] Add DummyClassifier baseline.
* [ ] Add Logistic Regression.
* [ ] Add SVM.
* [ ] Add KNN.
* [ ] Add Random Forest.
* [ ] Compare all models fairly.
* [ ] Add Stratified K-Fold cross-validation.

## Priority 5 — Improve neural-network training

* [ ] Add EarlyStopping.
* [ ] Set random seeds.
* [ ] Experiment with batch size if necessary.
* [ ] Avoid increasing architecture complexity unless justified.

## Priority 6 — GitHub cleanup

* [ ] Clean debugging cells.
* [ ] Improve markdown explanations.
* [ ] Improve comments.
* [ ] Add dataset source.
* [ ] Add `README.md`.
* [ ] Add `requirements.txt`.
* [ ] Add `.gitignore`.
* [ ] Document final results.
* [ ] Document limitations.

---

# 41. Suggested Final Evaluation Strategy

A more professional evaluation process would be:

```text
1. Keep the final test set untouched.

2. Use training/validation data for:
   - preprocessing decisions
   - model selection
   - class-weight experiments
   - threshold selection
   - hyperparameter tuning

3. Select the final model.

4. Evaluate it once on the test set.

5. Report:
   - Accuracy
   - Precision
   - Recall
   - F1
   - Balanced Accuracy
   - ROC-AUC
   - Confusion Matrix

6. Discuss both strengths and limitations.
```

---

# 42. Important Lesson From the Current Experiment

The most valuable result from this project is not simply:

```text
Accuracy ≈ 81%
```

The deeper lesson is:

```text
High accuracy does not necessarily mean
good classification performance.
```

The current model performs extremely well on the majority class:

```text
Class 0 Recall ≈ 98%
```

but performs poorly at detecting the minority class:

```text
Class 1 Recall ≈ 31%
```

Therefore:

```text
Accuracy alone hides an important weakness.
```

This experiment demonstrates why imbalanced classification requires metrics such as:

```text
Recall
F1-score
Balanced Accuracy
Confusion Matrix
ROC-AUC
Precision-Recall analysis
```

This is an important machine-learning lesson and should remain documented in the project.

---

# 43. Long-Term Goal

The objective of future revisions should not be to simply make the neural network larger or obtain the highest possible accuracy.

The real objective should be to build a **correct, reproducible and interpretable machine learning experiment**.

The project should eventually answer:

```text
Which model performs best?

Why?

How much better is it than a baseline?

How well does it identify the minority class?

Is the result stable across different data splits?

What trade-off exists between precision and recall?

Can the experiment be reproduced by another developer?
```

Once these questions are answered, the project will represent much more than a basic TensorFlow notebook.

It will demonstrate a complete machine-learning workflow.

---

# 44. Revision Notes

### Current uploaded notebook

```text
Version: Initial Neural Network Experiment
Dataset: Haberman Survival Dataset
Samples: 306
Features: 3
Model: Dense Neural Network
Hidden Units: 10
Output: Sigmoid
Loss: Binary Cross-Entropy
Optimizer: Adam
Epochs: 200
Batch Size: 20
Observed Accuracy: ~81%
```

### Findings obtained after the uploaded notebook

```text
Class 0 Recall: 0.98
Class 1 Recall: 0.31
Class 1 F1-score: 0.45

Confusion Matrix:
[[45, 1],
 [11, 5]]
```

Calculated balanced class weights:

```text
Class 0: ~0.6816
Class 1: ~1.8769
```

These experiments should be incorporated into the next notebook revision.

---

# 45. Next Revision Target

The immediate next milestone should be:

```text
Version 2
│
├── Proper train/validation/test separation
├── StandardScaler
├── Correct evaluation metrics
├── Baseline classifier
├── Class weights
├── Early stopping
├── Threshold analysis
├── Classical ML comparison
└── Final model comparison
```

After completing these steps, the project will be ready for a substantially cleaner and more meaningful GitHub release.
