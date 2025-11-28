# Deep Learning – Atelier 1  
## Regression & Multi-Class Classification  
### Université Abdelmalek Essaâdi – FST Tanger  
### Master MBD – Deep Learning Lab 1

---

## Overview

This repository contains two complete deep learning projects implemented during **Atelier 1**:

1. **Part 1 — Regression Task**  
   Predict continuous stock-related values from the **NYSE dataset** using a Deep Neural Network (MLP).

2. **Part 2 — Multi-Class Classification Task**  
   Predict machine failure types using the **Predictive Maintenance dataset**.

Each part includes:

- Data preprocessing  
- Exploratory Data Analysis (EDA)  
- Neural network architecture (PyTorch)  
- Hyperparameter tuning  
- Regularization techniques  
- Loss and accuracy visualizations  
- Final evaluation & interpretation

---

# Part 1 — Regression (NYSE Stock Dataset)

### Dataset  
Kaggle: https://www.kaggle.com/datasets/dgawlik/nyse

The dataset includes multiple stock symbols and historical daily values such as opening price, closing price, volume, etc.

---

## 1. Exploratory Data Analysis (EDA)

Performed:

- Handling missing values  
- Chronological sorting of time-series data  
- Statistical summaries  
- Feature distributions  
- Correlation heatmaps  
- Symbol frequency exploration  

### Symbol Encoding  
Each stock symbol was converted to an integer ID:

```python
df['symbol_id'] = df['symbol'].astype('category').cat.codes
```

## 2. Data Preprocessing

- Chronological train/validation/test split
- Normalization using StandardScaler
- Removal of unused features
- Prevention of data leakage
- Tensors moved to GPU when available

## 3. Model Architecture (MLP)

```bash
class StockRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)
```

## 4. Hyperparameter Tuning:

Using GridSearchCV with skorch:
Parameters explored:
- Learning rate
- Hidden layer sizes
- Optimizer (SGD / Adam / RMSprop)
- Epoch count
- Batch size
- Activation functions

## 5. Results:

Loss curves for training and validation:
<img width="399" height="374" alt="Untitled" src="https://github.com/user-attachments/assets/a98e59ae-cce7-442a-bed4-40867e75e37b" />

Observations:
- Smooth convergence
- No divergence or exploding gradients
- Low generalization gap after regularization

## 6. Regularization Techniques

- Dropout
- L2 weight decay
- Early stopping
- Batch Normalization
These improved the model's stability and validation performance.

