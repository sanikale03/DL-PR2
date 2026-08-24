# 📄 Summary Report

# 🧠 Deep Learning PR-2 — Adult Income Classification

---

## 📌 Business Problem

Income classification is a **binary classification** problem where the objective is to predict whether an individual's annual income is **greater than 50K** based on demographic, employment, education and financial attributes.

The dataset contains a mixture of numerical and categorical features. Because the target is imbalanced, evaluating the model using accuracy alone can be misleading.

The objective of this project is to build and compare deep learning models while analysing **accuracy, precision, recall, F1-score and ROC-AUC**, with particular attention to the minority class `>50K`.

---

# 📂 Dataset Description

The project uses the **Adult Income / Census Income dataset** supplied as `adult.csv`.

### Dataset Details

- Original Records: **48,842**
- Original Features/Columns: **15**
- Problem: **Binary Classification**
- Target Column: `income`

### Target

```text
0 → <=50K
1 → >50K
```

### Important Features

The dataset contains:

- `age`
- `workclass`
- `fnlwgt`
- `education`
- `educational-num`
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `gender`
- `capital-gain`
- `capital-loss`
- `hours-per-week`
- `native-country`

---

# ⚙️ Data Preprocessing

Performed:

- Dataset loading
- Shape verification
- Data type inspection
- Whitespace stripping
- Replacement of `?` with `NaN`
- Missing-value analysis
- Removal of rows containing missing values
- Removal of `fnlwgt`
- Removal of text `education`
- Target encoding
- Class distribution analysis
- One-Hot Encoding
- Stratified 80/20 train-test split
- `random_state=42`
- StandardScaler fitted only on training data

### Actual Dataset After Cleaning

```text
Original rows       : 48,842
Rows after dropna   : 45,222
Columns after drops : 13
Encoded input       : 80 features
```

The actual uploaded `adult.csv` is used as the source of truth for the experiments.

### Scaling

Numeric features are standardized:

```text
age
educational-num
capital-gain
capital-loss
hours-per-week
```

The scaler is fitted only on `X_train` and then applied to both training and test data.

This prevents test-set information leakage.

---

# 📊 Exploratory Data Analysis

The project performs:

- Class distribution analysis
- Age distribution by income
- Hours-per-week analysis
- Education-level analysis
- Numeric correlation analysis

### Class Distribution

```text
<=50K → Majority Class
>50K  → Minority Class
```

The class imbalance makes **Precision, Recall and F1-score for class 1** important evaluation metrics.

---

# 🤖 Models & Experiments Implemented

## 1. Baseline Multi-Layer Perceptron

```text
80 Input Features
        ↓
Dense(128, ReLU)
        ↓
Dense(64, ReLU)
        ↓
Dense(1, Sigmoid)
```

Configuration:

- ReLU activation
- Glorot Uniform initialization
- Adam optimizer
- Binary Crossentropy
- 50 epochs
- Batch size = 64

### Parameters

For the actual 80 encoded input features:

```text
80 × 128 + 128 = 10,368

128 × 64 + 64 = 8,256

64 × 1 + 1 = 65

Total = 18,689 parameters
```

The baseline acts as the reference model for all subsequent experiments.

---

# 🧠 2. Activation Function Comparison

Compared hidden-layer activation functions:

- ReLU
- Tanh
- Sigmoid
- ELU

### ReLU

- Fast and computationally efficient
- Commonly used in hidden layers
- Strong gradient flow for positive inputs
- Can suffer from dead neurons

### Tanh

- Output range from `-1` to `+1`
- Zero-centred
- Can suffer from vanishing gradients when saturated

### Sigmoid

- Output range from `0` to `1`
- Suitable for binary classification output
- Can suffer from vanishing gradients in hidden layers

### ELU

- Allows negative outputs
- Provides smoother behavior for negative inputs
- Can reduce dead-neuron behavior compared with ReLU

The notebook compares validation accuracy and minority-class F1-score for all four activations.

---

# 🔎 ReLU Dead-Neuron Analysis

The first hidden ReLU layer is inspected to identify neurons producing zero activations.

A dead-neuron fraction is calculated using:

```python
dead_fraction = np.mean(activations == 0)
```

This demonstrates how ReLU neurons can become inactive for a large portion of the input.

---

# 📉 Sigmoid Gradient-Flow Analysis

Gradient magnitudes are measured for:

```text
Layer 1
Layer 2
Output Layer
```

Sigmoid hidden layers can suffer from vanishing gradients because:

```text
σ'(x) = σ(x)(1 - σ(x))
```

and the maximum derivative is only `0.25`.

When sigmoid units saturate, gradients can become very small.

---

# ⚖️ 3. Weight Initialization

Compared:

- Glorot Uniform
- Glorot Normal
- He Uniform
- He Normal
- Zeros

### Glorot Initialization

```text
Var(W) ≈ 2 / (fan_in + fan_out)
```

Glorot initialization helps maintain stable activation and gradient variance.

### He Initialization

```text
Var(W) ≈ 2 / fan_in
```

He initialization is particularly appropriate for ReLU-based networks.

### Zero Initialization

All hidden neurons start with identical weights.

This creates a **symmetry problem**:

```text
Same weights
      ↓
Same outputs
      ↓
Same gradients
      ↓
Same updates
      ↓
Neurons remain identical
```

Therefore, zero initialization prevents the hidden layer from learning diverse representations.

---

# 📈 4. Loss Functions

Compared:

- Binary Crossentropy
- Mean Squared Error
- Weighted Binary Crossentropy
- Focal Loss

## Binary Crossentropy

```text
L = -[y log(p) + (1-y) log(1-p)]
```

BCE is the natural loss for a binary classifier with a sigmoid output.

## MSE

MSE is primarily a regression loss but is included as a classification comparison experiment.

## Weighted BCE

Class weights are calculated using:

```python
compute_class_weight(
    class_weight="balanced",
    classes=[0, 1],
    y=y_train
)
```

Weighted BCE gives greater importance to minority-class examples.

## Focal Loss

Focal Loss is implemented as a custom Keras loss.

Configuration:

```text
Gamma = 2.0
Alpha = 0.25
```

It reduces the influence of easy examples and focuses training more strongly on difficult examples.

---

# 🧮 5. Batch Normalization

The Batch Normalization architecture is:

```text
Dense(128)
      ↓
BatchNorm
      ↓
ReLU
      ↓
Dense(64)
      ↓
BatchNorm
      ↓
ReLU
      ↓
Dense(1, Sigmoid)
```

### Batch Normalization Formula

```text
BN(x) = γ((x - μB) / √(σB² + ε)) + β
```

where:

```text
γ → Learned scale
β → Learned shift
```

### Benefits

- More stable training
- Faster convergence
- Reduced sensitivity to learning rate
- Mild regularization effect

---

# 🔄 BatchNorm Position Experiment

Compared:

```text
Dense → BatchNorm → ReLU
```

with:

```text
Dense → ReLU → BatchNorm
```

The validation performance is compared using the same dataset and training setup.

A paired bootstrap confidence interval is also calculated for the validation accuracy difference.

---

# 📊 BatchNorm Gamma and Beta

The trained BatchNorm layer is inspected to analyse:

- Gamma values
- Beta values
- Running mean
- Running variance

A gamma value close to zero can strongly suppress the corresponding normalized feature.

---

# ⚡ 6. Optimizer Comparison

Compared:

- SGD
- SGD + Momentum
- RMSprop
- Adam
- Explicit Adam

## SGD

Uses a global learning rate:

```text
θ = θ - learning_rate × gradient
```

## SGD + Momentum

Momentum accumulates previous gradient information and can reduce oscillations.

## RMSprop

Uses a moving average of squared gradients to adapt parameter updates.

## Adam

Adam combines:

- First-moment estimation
- Second-moment estimation
- Bias correction

## Explicit Adam

```python
Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999
)
```

The notebook compares validation accuracy and class-1 F1 across the optimizers.

---

# 🎚️ 7. Learning Rate Sensitivity

### SGD

Compared:

```text
0.0001
0.001
0.01
0.1
```

### Adam

Compared:

```text
0.0001
0.001
0.01
```

The experiment demonstrates how learning rate affects convergence speed and training stability.

---

# 🏆 Final Combined Model

The final model combines the main techniques found to be useful during the experiments.

```text
80 Input Features
        ↓
Dense(128, He Normal)
        ↓
BatchNorm
        ↓
ReLU
        ↓
Dense(64, He Normal)
        ↓
BatchNorm
        ↓
ReLU
        ↓
Dense(1, Sigmoid)
```

### Final Configuration

```text
Activation     : ReLU
Initializer    : He Normal
BatchNorm      : Yes
Optimizer      : Adam
Loss           : BCE / Weighted BCE
Epochs         : 80
Batch Size     : 64
```

Weighted BCE is selected automatically when it provides a higher minority-class F1-score than the baseline BCE model.

---

# 📊 Model Evaluation

The complete comparison includes:

| Model | Activation | Initialization | Loss | BatchNorm | Optimizer | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| Baseline | ReLU | Glorot Uniform | BCE | No | Adam | Generated | Generated | Generated | Generated | Generated |
| Best Activation | Selected | Glorot Uniform | BCE | No | Adam | Generated | Generated | Generated | Generated | Generated |
| Best Initializer | ReLU | Selected | BCE | No | Adam | Generated | Generated | Generated | Generated | Generated |
| Weighted BCE | ReLU | He Normal | Weighted BCE | No | Adam | Generated | Generated | Generated | Generated | Generated |
| Focal Loss | ReLU | He Normal | Focal Loss | No | Adam | Generated | Generated | Generated | Generated | Generated |
| BatchNorm | ReLU | He Normal | BCE | Yes | Adam | Generated | Generated | Generated | Generated | Generated |
| Final Combined | ReLU | He Normal | Selected | Yes | Adam | Generated | Generated | Generated | Generated | Generated |

**Important:** Run `DL_PR2.ipynb` first and use the actual generated metrics in the final submission.

---

# 📈 ROC-AUC Analysis

The notebook generates ROC curves for:

- Baseline ANN
- Weighted BCE
- Final Combined Model

ROC-AUC evaluates the ability of the classifier to distinguish the two income classes across different probability thresholds.

---

# 💼 Recommendation

The **Final Combined Model** is the assignment's final candidate because it combines:

- ReLU activation
- He Normal initialization
- Batch Normalization
- Adam optimization
- BCE or Weighted BCE based on measured minority-class F1

The final recommendation should be confirmed using the actual executed test results, especially:

- Precision
- Recall
- F1-score
- ROC-AUC

Because the `>50K` class is the minority class, **minority-class F1 and Recall should receive strong consideration rather than selecting a model using accuracy alone**.

This project is an academic machine-learning project and is not intended as a standalone financial or employment decision-making system.

---

# 🚀 Future Improvements

- Hyperparameter tuning
- Cross-validation
- ROC-AUC optimization
- Precision-Recall AUC
- Validation-based threshold optimization
- Feature importance analysis
- SHAP explainability
- Probability calibration
- External dataset validation
- Model deployment using Flask/FastAPI
- Web application deployment
- Continuous model monitoring
- Fairness and bias analysis

---

# ✅ Conclusion

This project demonstrates a complete deep learning workflow for binary classification using the **Adult Income / Census Income dataset**.

The work progresses from a baseline MLP to systematic experiments involving:

- Activation functions
- Weight initialization
- Loss functions
- Batch Normalization
- Optimizers
- Learning-rate sensitivity

The final combined ANN integrates **ReLU + He Normal + Batch Normalization + Adam** and selects BCE or Weighted BCE based on minority-class F1 performance.

The project demonstrates that an imbalanced classification problem should not be evaluated using accuracy alone. **Precision, Recall, F1-score and ROC-AUC provide a more complete view of model performance.**

---

# 👩‍💻 Author

**Sanika Kale**

MCA Student | Data Analytics & Machine Learning

**Red & White Skill Education**
