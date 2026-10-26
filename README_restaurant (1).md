# Restaurant Monthly Revenue Prediction

A supervised machine learning **regression** project that predicts a restaurant's monthly revenue from business factors such as customer numbers, pricing and marketing spend. Built with Python, Pandas and scikit-learn.

> **Status:** work in progress.

## Dataset

- **File:** `data/restaurant_data.csv`
- **Size:** 1,000 rows and 8 columns, with no missing values and no duplicate rows
- **Target:** `Monthly_Revenue`

| Feature | Description |
| --- | --- |
| `Number_of_Customers` | Customers served |
| `Menu_Price` | Menu price level |
| `Marketing_Spend` | Marketing spending |
| `Cuisine_Type` | Japanese, American, Mexican or Italian |
| `Average_Customer_Spending` | Average spend per customer |
| `Promotions` | Promotion indicator |
| `Reviews` | Review score |

Cuisine types are fairly balanced: Japanese (262), American (256), Mexican (250) and Italian (232).

## Workflow

1. **Data checks:** `head()`, `info()`, category counts and a duplicate check.
2. **Correlation analysis:** correlation of each feature with `Monthly_Revenue`.
3. **Feature preparation:** `Cuisine_Type` mapped to numbers (Japanese 0, American 1, Mexican 2, Italian 3).
4. **Train/test split:** 80% training and 20% testing, with `random_state=42`.
5. **Modelling:**
   - Decision Tree Regressor in a scikit-learn `Pipeline`, tuned with `GridSearchCV` (5-fold cross-validation, scored on R²) over `max_depth`, `min_samples_split` and `min_samples_leaf`
   - Linear Regression
6. **Evaluation:** R², MSE and RMSE on the test set.

## Key findings

Correlation with monthly revenue:

| Feature | Correlation |
| --- | --- |
| Number_of_Customers | 0.75 |
| Marketing_Spend | 0.27 |
| Menu_Price | 0.26 |
| Promotions | 0.01 |
| Reviews | -0.02 |
| Average_Customer_Spending | -0.04 |

The number of customers is by far the strongest predictor of revenue.

## Results

| Model | R² | RMSE |
| --- | --- | --- |
| Decision Tree Regressor (tuned) | 0.58 | 68.09 |
| Linear Regression | 0.67 | 59.74 |

Linear Regression performed better than the tuned Decision Tree on this dataset, explaining about 67% of the variation in monthly revenue.

## Tech stack

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## How to run

1. Clone the repository:
   ```bash
   git clone https://github.com/carinaigbodor/<repo-name>.git
   cd <repo-name>
   ```
2. Install the dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn jupyter
   ```
3. Make sure `restaurant_data.csv` is inside a `data/` folder.
4. Open the notebook and run all cells:
   ```bash
   jupyter notebook Restaurant_Prediction_Model.ipynb
   ```

## Next steps

- One-hot encode `Cuisine_Type` instead of numbering it, so the models don't treat cuisines as ordered.
- Try Random Forest and other regressors, and compare them with cross-validation.
- Plot actual vs predicted revenue and look at the residuals.
- Report which features each model relies on most.
- Investigate why the weaker features (reviews, promotions, average spend) add so little.

## Author

**Carina Uregbo Igbodor**
AI/ML Engineering Fellow, Women in AI Fellowship
[LinkedIn](https://linkedin.com/in/carina-igbodor-753690195) | carinaigbodor0@gmail.com
