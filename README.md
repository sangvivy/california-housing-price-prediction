# California Housing Price Prediction

## 1. Project Overview

This project develops a machine learning model to predict median house values in California using the California Housing dataset from Scikit-learn.

The project uses a **K-Nearest Neighbors (KNN) Regressor** within a Scikit-learn Pipeline. Data preprocessing is handled using `ColumnTransformer`, while `GridSearchCV` is used to identify the best hyperparameters through 5-fold cross-validation.

A Streamlit application is also developed to allow users to enter housing characteristics and obtain predictions using the trained model.

## Streamlit Application Preview

The project includes an interactive Streamlit application
that allows users to input housing characteristics and
generate median house value predictions.

![California Housing Prediction Dashboard](images/streamlit_dashboard.png)

### Project Objectives

- Load and explore the California Housing dataset.
- Apply appropriate preprocessing techniques using `ColumnTransformer`.
- Split the dataset into training and testing sets using an 80:20 ratio.
- Develop a K-Nearest Neighbors regression model.
- Optimize model hyperparameters using `GridSearchCV`.
- Evaluate model performance using the R² score.
- Save the trained machine learning pipeline using Joblib.
- Develop an interactive Streamlit application for making predictions.

---

## 2. Dataset Description

The project uses the **California Housing dataset**, which is available through Scikit-learn's `fetch_california_housing` function.

The dataset contains information about housing districts in California, with observations collected from the 1990 U.S. Census.

### Dataset Features

The dataset contains eight independent variables:

| Feature | Description |
|---|---|
| MedInc | Median income in the block group |
| HouseAge | Median house age |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | Block group population |
| AveOccup | Average number of household members |
| Latitude | Geographic latitude |
| Longitude | Geographic longitude |

**Target Variable:**

- `MedHouseVal`: Median house value for the block group, expressed in units of $100,000.

### Loading the Dataset

The dataset is loaded directly from Scikit-learn:

```python
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target
```

No separate CSV download is required.

---

## 3. Technologies and Libraries

The project uses the following technologies:

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing, model development and evaluation |
| Joblib | Saving and loading the trained pipeline |
| Streamlit | Interactive web application |
| Matplotlib | Data visualization |
| Jupyter Notebook | Model development and experimentation |
| VS Code | Development environment |
| Git and GitHub | Version control and project hosting |

---

## 4. Project Workflow

The machine learning workflow follows these stages:

1. Dataset loading and exploration.
2. Data preprocessing.
3. Train-test splitting.
4. Machine learning pipeline development.
5. Hyperparameter tuning using GridSearchCV.
6. Model evaluation.
7. Model serialization.
8. Streamlit application development.

### 4.1 Data Preprocessing

A `ColumnTransformer` is used to organize the preprocessing operations.

`StandardScaler` is applied to standardize the numerical features before training the KNN model.

Standardization is important because KNN uses distance calculations to identify neighbouring observations. Features with larger numerical scales could otherwise have a disproportionate influence on those calculations.

### 4.2 Train-Test Split

The dataset is divided into:

- 80% training data.
- 20% testing data.

The training set is used for model development and hyperparameter optimization, while the test set is reserved for evaluating the final selected model.

### 4.3 Machine Learning Pipeline

A Scikit-learn Pipeline combines preprocessing and model training into one workflow.

The pipeline consists of:

1. A preprocessing stage using `ColumnTransformer`.
2. A regression stage using `KNeighborsRegressor`.

This ensures that the same preprocessing operations are applied consistently during training and prediction.

It also helps prevent data leakage because preprocessing is fitted within the cross-validation training folds.

### 4.4 Hyperparameter Tuning

`GridSearchCV` is used to identify the best combination of KNN hyperparameters.

The following parameter grid is evaluated:

| Hyperparameter | Values |
|---|---|
| n_neighbors | [3, 5, 7, 9] |
| weights | ['uniform', 'distance'] |
| p | [1, 2] |

The search evaluates all 16 possible combinations using 5-fold cross-validation.

The R² score is used as the optimization metric.

### 4.5 Model Evaluation

The best-performing model is evaluated on the held-out test dataset using the R² score.

R² measures how much of the variation in the target variable is explained by the model.

---

## 5. Model Results

The hyperparameter optimization process identified the following best parameters:

```python
{
    'model__n_neighbors': 9,
    'model__p': 1,
    'model__weights': 'distance'
}
```

### Best Hyperparameters Explained

- **n_neighbors = 9:** The model considers nine neighbouring observations when making predictions.
- **weights = distance:** Closer neighbours have greater influence on the prediction.
- **p = 1:** Manhattan distance is used to calculate distances between observations.

### Evaluation Results

| Evaluation Metric | Score |
|---|---:|
| Best Cross-Validation R² | 0.7313 |
| Test R² | 0.7221 |

The model achieved a cross-validation R² score of approximately 0.7313 and a test R² score of approximately 0.7221.

The relatively small difference between the two scores suggests that the model performs consistently across the validation folds and the held-out test dataset.

The test R² indicates that the model explains approximately 72.21% of the variation in median housing values in the test dataset.

---

## 6. Model Serialization

After hyperparameter optimization and evaluation, the trained pipeline is saved using Joblib.

The complete pipeline, including preprocessing and the trained KNN model, is serialized into a `.pkl` file.

The saved model is located at:

```text
models/california_housing_model.pkl
```

Saving the complete pipeline allows the Streamlit application to load the trained model and make predictions without repeating the training process.

---

## 7. Streamlit Application

An interactive Streamlit application is developed to demonstrate the trained model.

The application allows users to:

- Enter housing characteristics.
- Provide values for the eight independent variables.
- Submit the information for prediction.
- View the predicted median house value.

The application loads the previously saved machine learning pipeline using Joblib.

This separates model development from application usage and makes it possible to use the trained model without retraining every time the application starts.

### Running the Streamlit Application

From the project directory, activate the virtual environment and execute:

```bash
python -m streamlit run app.py
```

The application will open in a local web browser.

---

## 8. Project Structure

```text
California-housing-price-prediction/
│
├── .venv/
│
├── models/
│   └── california_housing_model.pkl
│
├── california_housing_prediction.ipynb
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

### File Descriptions

| File | Description |
|---|---|
| `california_housing_prediction.ipynb` | Contains dataset loading, preprocessing, model development, hyperparameter tuning and evaluation. |
| `app.py` | Contains the Streamlit application code. |
| `models/california_housing_model.pkl` | Stores the trained machine learning pipeline. |
| `requirements.txt` | Lists the Python dependencies required to run the project. |
| `.gitignore` | Specifies files and directories excluded from version control. |
| `README.md` | Provides project documentation and setup instructions. |

---

## 9. Installation and Setup

### Prerequisites

- Python 3.12
- Visual Studio Code or another Python development environment
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/sangvivy/California_Housing_Project.git
```

### Step 2: Navigate to the Project Directory

```bash
cd California_Housing_Project
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

### Step 5: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 6: Run the Notebook

Open `california_housing_prediction.ipynb` in VS Code, select the project's virtual environment as the Jupyter kernel, and execute the notebook cells.

### Step 7: Launch the Streamlit Application

```bash
python -m streamlit run app.py
```

**Note:** The saved model must be available in the `models` directory for the Streamlit application to load successfully. If it is not included in the downloaded repository, run the notebook to generate it.

---

## 10. Limitations

- The model is trained using California housing data from the 1990 U.S. Census and may not reflect current housing market conditions.
- The model is specific to the dataset's geographical and economic context.
- Predictions depend on the quality and representativeness of the input features.
- The model's performance is evaluated using a random train-test split, which does not necessarily represent performance across different geographical regions or time periods.

---

## 11. Conclusion

This project demonstrates the application of machine learning techniques to a regression problem using the California Housing dataset.

By combining `ColumnTransformer`, Scikit-learn Pipelines, `KNeighborsRegressor`, and `GridSearchCV`, the project establishes a structured workflow for preprocessing, model optimization and evaluation.

The selected model achieved a test R² score of approximately 0.7221.

The additional Streamlit application provides an interactive interface for using the trained pipeline to generate housing value predictions.

---

## 12. Repository

**GitHub:** https://github.com/sangvivy/California_Housing_Project

**Developed using:** Python, Scikit-learn and Streamlit.
