# House Price Prediction

Predicting residential house prices using regression models, built as part of the Xylofy AI Internship — Task 1.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Flow](#pipeline-flow)
- [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
- [Feature Engineering](#feature-engineering)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Building](#model-building)
- [How Each Model Works](#how-each-model-works)
- [Model Results](#model-results)
- [Key Insights](#key-insights)
- [Visualizations](#visualizations)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)

---

## Overview

This project builds and evaluates machine learning regression models to predict residential house prices from structural features, amenities, and furnishing status. The goal is to identify the strongest price drivers and compare a simple linear baseline against a more complex ensemble model.

**Dataset size:** 545 records, 13 raw attributes
**Models trained:** Linear Regression, Random Forest Regressor
**Best model:** Linear Regression (R² = 0.6627)

---

## Dataset

The Housing dataset contains the following columns:

| Column | Type | Description |
|---|---|---|
| `price` | Numeric (target) | House price in INR |
| `area` | Numeric | Plot area in sq. ft. |
| `bedrooms` | Numeric | Number of bedrooms |
| `bathrooms` | Numeric | Number of bathrooms |
| `stories` | Numeric | Number of floors |
| `parking` | Numeric | Number of parking spots |
| `mainroad` | Binary (yes/no) | Main road access |
| `guestroom` | Binary (yes/no) | Guest room present |
| `basement` | Binary (yes/no) | Basement present |
| `hotwaterheating` | Binary (yes/no) | Hot water heating present |
| `airconditioning` | Binary (yes/no) | Air conditioning present |
| `prefarea` | Binary (yes/no) | Located in a preferred area |
| `furnishingstatus` | Categorical | furnished / semi-furnished / unfurnished |

No missing values or duplicate rows were found in the raw dataset.

## Pipeline Flow

```
Raw CSV
   |
   v
Data Loading & Initial Inspection (head, tail, sample, info, describe)
   |
   v
Exploratory Data Analysis
   |-- Missing value check
   |-- Duplicate check & removal
   |-- Univariate analysis (numerical + categorical)
   |-- Target variable analysis (price distribution)
   |-- Bivariate analysis (price vs. each feature)
   |-- Categorical analysis (price by amenity)
   |-- Correlation analysis
  `-- Outlier detection
   |
   v
Feature Engineering
   |-- TotalRooms = bedrooms + bathrooms
   |-- AreaPerBedroom = area / bedrooms
   |-- AreaPerBathroom = area / bathrooms
   |-- AmenityScore = count of yes-valued binary amenities
   -- LuxuryScore = AmenityScore + furnished_flag + parking_flag
   |
   v
Encoding
   |-- One-hot encoding on binary yes/no columns
   -- One-hot encoding on furnishingstatus
   |
   v
Feature Selection
   |
   v
Train-Test Split (80/20, random_state=42)
   |
   v
Model Training
   |-- Linear Regression
   -- Random Forest Regressor (n_estimators=200)
   |
   v
Evaluation (MAE, RMSE, R^2)
   |
   v
Visualization
   |-- Model comparison chart
   |-- Feature importance (Random Forest)
   |-- Actual vs. Predicted scatter plot
   -- Residual analysis
   |
   v
Insights & Conclusion
```

---

## Data Cleaning & Preprocessing

- **Missing values:** Checked across all 13 columns — none found, so no imputation was required.
- **Duplicates:** Checked using `df.duplicated()` — none found.
- **Encoding:** All categorical columns (6 binary yes/no columns and `furnishingstatus`) were converted to numeric form using **one-hot encoding** via `pd.get_dummies()`, with `drop_first=True` to avoid the dummy variable trap.
- **Feature selection:** All engineered and encoded columns were retained, since each showed non-trivial correlation with price or carried meaningful structural/comfort information.

---

## Feature Engineering

Five new features were constructed to capture compound signals not present in the raw columns:

| Feature | Formula | Purpose |
|---|---|---|
| `TotalRooms` | `bedrooms + bathrooms` | Overall room count |
| `AreaPerBedroom` | `area / bedrooms` | Space efficiency per bedroom |
| `AreaPerBathroom` | `area / bathrooms` | Space efficiency per bathroom |
| `AmenityScore` | count of yes-valued binary amenities (0–6) | Overall amenity richness |
| `LuxuryScore` | `AmenityScore + furnished_flag + parking_flag` | Composite luxury indicator |

These engineered features were retained in the final feature set and turned out to be highly predictive — `LuxuryScore` and `TotalRooms` ranked 2nd and 3rd in Random Forest feature importance, just behind `area`.

---

## Exploratory Data Analysis

Key findings from EDA:

- **Price distribution** is right-skewed — most homes fall between ₹20L–₹60L, with a smaller set of high-end outliers above ₹90L. These outliers were retained as legitimate premium listings rather than treated as errors.
- **Area** (r = 0.54) and **bathrooms** (r = 0.52) are the strongest linear predictors of price.
- Houses with **main road access**, **air conditioning**, or location in a **preferred area** show consistently higher median prices.
- **Furnished** homes command higher prices than semi-furnished or unfurnished homes.
- Outliers in `area` and `price` correspond to valid large/premium properties, not data entry errors, and were kept in the dataset given its small size (545 rows).

---

## Model Building

- **Split:** 80% train / 20% test, `random_state=42` for reproducibility.
- **Models trained:**
  - `LinearRegression()` — scikit-learn defaults, ordinary least squares, no regularization.
  - `RandomForestRegressor(n_estimators=200, random_state=42)` — all other hyperparameters left at default (no `max_depth` limit, no `min_samples_leaf` tuning).
- **Hyperparameter tuning:** No formal `GridSearchCV` / `RandomizedSearchCV` was performed. `n_estimators=200` was chosen as a standard baseline sufficient for this dataset's size, since deeper tuning offered diminishing returns on only 545 rows.
- **Metrics used:** MAE, RMSE, R² Score.

---

## How Each Model Works

### Linear Regression

Linear Regression assumes price is a weighted sum of the input features:
price = w1·area + w2·bathrooms + w3·stories + ... + bias

It solves a closed-form equation (ordinary least squares) that finds the exact weights minimizing the squared difference between predicted and actual prices across all training rows simultaneously — no iteration, no randomness, fully deterministic. This worked well here because price in this dataset increases fairly steadily with area, bathrooms, and stories, matching the model's linear assumption.

### Random Forest Regressor

Random Forest trains 200 separate decision trees. Each tree:

1. Sees a random bootstrap sample of training rows.
2. At each split, considers a random subset of features.
3. Repeatedly splits the data on feature thresholds (e.g., "is area > 5000?") until each leaf contains a small group of similar houses, predicting their average price.

The final prediction averages all 200 trees' outputs, which smooths out individual tree errors. Random Forest is designed to capture non-linear relationships and feature interactions, but with only 436 training rows split across 200 trees, each tree had too little data to learn genuinely useful non-linear patterns without overfitting — which is why it didn't outperform the simpler linear model here.

---

## Model Results

| Model | MAE (Lakhs) | RMSE (Lakhs) | R² |
|---|---|---|---|
| **Linear Regression** | 9.63 | 13.06 | **0.6627** |
| Random Forest | 9.86 | 13.64 | 0.6317 |

**In plain terms:** the Linear Regression model's predictions were off by about ₹9.6 lakh on average, and it explained roughly 66% of the variation in house prices.

Linear Regression slightly outperformed Random Forest across all three metrics — consistent with the dataset's largely linear, additive relationship between features and price at this scale.

---

## Key Insights

- **Area is the single strongest price driver** (~38% importance in Random Forest), followed by the engineered `LuxuryScore` and `TotalRooms` features — confirming the value of domain-driven feature construction.
- **Residuals** were reasonably centered around zero with no strong systematic bias, though both models underpredicted high-value properties above ₹80L.
- **Random Forest did not outperform Linear Regression** — price behaves in a largely additive, linear way at this data scale rather than depending on complex feature interactions. This is consistent with Random Forest's general tendency to need larger datasets before its added flexibility pays off.
- **Surprise finding:** the more complex model losing to the simpler one is a reminder that model complexity should match data scale — bigger isn't always better.

---

## Visualizations

All charts are saved under `charts/` as PNG files:

| Chart | File |
|---|---|
| Missing values heatmap | `01_missing_values_heatmap.png` |
| Numerical feature distributions | `02_numerical_distributions.png` |
| Categorical feature counts | `03_categorical_counts.png` |
| Price distribution (histogram, KDE, boxplot) | `04_price_distribution.png` |
| Price vs. numerical features | `05_price_vs_numerical.png` |
| Price by binary features | `06_price_by_binary_features.png` |
| Price by furnishing status | `07_price_by_furnishing.png` |
| Correlation heatmap | `08_correlation_heatmap.png` |
| Outlier boxplots | `09_outlier_boxplots.png` |
| Model comparison (MAE / RMSE / R²) | `10_model_comparison.png` |
| Feature importance (Random Forest) | `11_feature_importance.png` |
| Actual vs. Predicted price | `12_actual_vs_predicted.png` |
| Residual analysis | `13_residual_analysis.png` |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Jupyter Notebook | Development environment |
| Pandas | Data loading & cleaning |
| NumPy | Numerical operations |
| Scikit-learn | Regression models & evaluation metrics |
| Matplotlib / Seaborn | Visualization |

## Project Structure
```
HousePricePrediction_BandiSreesaicharan/
├── analysis.ipynb        
├── Housing.csv            
├── summary.docx            
├── charts/              
   ├── 01_missing_values_heatmap.png
   ├── 02_numerical_distributions.png
   ├── ...
   └── 13_residual_analysis.png
```
