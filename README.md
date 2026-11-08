# California Housing Price Prediction

An end-to-end regression project using the California Housing dataset.

## Project Status

The California Housing Price Prediction model has been
developed using a Scikit-learn Pipeline and GridSearchCV.
The model is saved using Joblib for future predictions.

## Workflow
- Dataset loading and exploration
- Numerical preprocessing with `ColumnTransformer` and `StandardScaler`
- 80/20 train-test split
- KNN regression pipeline
- `GridSearchCV` with 5-fold cross-validation and R² scoring
- Test-set evaluation
- Model persistence with Joblib
- Streamlit prediction application

## Hyperparameter search
- `n_neighbors`: 3, 5, 7, 9
- `weights`: uniform, distance
- `p`: 1, 2

This evaluates 16 parameter combinations across 5 folds (80 CV fits).

## Run locally
Create and activate a virtual environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Open `notebooks/california_housing_prediction.ipynb` in VS Code and run the cells in order.

After training creates `models/california_housing_model.pkl`, launch the app:

```bash
streamlit run app.py
```

## Target
`MedHouseVal` is measured in units of $100,000. The app converts the model output to dollars for display.

## Note
Predictions are estimates based on historical California census data, not formal property valuations.
