import pandas as pd
import numpy as np
import joblib
import gc
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skopt import BayesSearchCV
import torch

# Check CUDA availability for GPU acceleration
print(torch.cuda.is_available())

# Set random seed for reproducibility
import random

SEED = 42  # You can change this to any fixed number

random.seed(SEED)
np.random.seed(SEED)
xgb.set_config(verbosity=0)  # Suppress XGBoost warnings
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(torch.cuda.is_available())


# Load Training and Validation Data
def load_data(train_path_placeholder, val_path_placeholder):
    """
    Loads training and validation data from CSV files.

    Args:
        train_path_placeholder (str): Placeholder for training dataset CSV path.
        val_path_placeholder (str): Placeholder for validation dataset CSV path.

    Returns:
        tuple: train_data DataFrame, val_data DataFrame, val_baseline Series.
    """
    train_data = pd.read_csv(train_path_placeholder)
    val_data = pd.read_csv(val_path_placeholder)

    # Replace zeros in RGB columns to 1 to avoid division by zero in later calculations
    train_data[['R', 'G', 'B']] = train_data[['R', 'G', 'B']].replace(0, 1)
    # Replace zero concentrations with a small positive number to avoid zero baseline
    train_data['Concentration'] = train_data['Concentration'].replace(0, 1e-6)

    # Use the first row of validation data as baseline
    val_baseline = val_data.iloc[0]
    # Use remaining rows as actual validation samples
    val_data = val_data.iloc[1:].reset_index(drop=True)

    return train_data, val_data, val_baseline


# Process Data for Training
def process_data(train_data):
    """
    Processes training data by calculating percent changes relative to baseline samples.

    Args:
        train_data (pd.DataFrame): Training dataset.

    Returns:
        tuple: Feature DataFrame X, target Series y.
    """
    print("Processing training data...")

    # Select first 500 samples sorted by concentration as baselines
    baseline_data = train_data.sort_values('Concentration').head(500)
    X_list, y_list = [], []

    # For each baseline, calculate features relative to it for entire dataset
    for _, baseline_row in baseline_data.iterrows():
        baseline_values = baseline_row[['R', 'G', 'B']]
        concentration_baseline = baseline_row['Concentration']

        # Percent change relative to baseline RGB values
        X_temp = (train_data[['R', 'G', 'B']] - baseline_values) / baseline_values * 100
        # Add average percent change across R,G,B
        X_temp['avg_pct_change'] = X_temp.mean(axis=1)
        # Add average absolute percent change across R,G,B
        X_temp['abs_avg_pct_change'] = X_temp[['R', 'G', 'B']].abs().mean(axis=1)

        # Append features and target differences
        X_list.append(X_temp.astype(np.float32))
        y_temp = train_data['Concentration'] - concentration_baseline
        y_list.append(y_temp.astype(np.float32))

    # Concatenate all feature and target data into single datasets
    return pd.concat(X_list, ignore_index=True), pd.concat(y_list, ignore_index=True)


# Load and process the dataset using placeholders for file paths
train_path = "path_to_train_dataset.csv"  # Replace with actual path
val_path = "path_to_validation_dataset.csv"  # Replace with actual path

train_data, val_data, val_baseline = load_data(train_path, val_path)
X, y = process_data(train_data)
del train_data  # Free memory


# Scale Data
y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))
y = pd.Series(y_scaled.flatten())
joblib.dump(y_scaler, "y_scaler.pkl")  # Save scaler for inverse transform later

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "X_scaler.pkl")  # Save feature scaler
del X  # Free memory


# Model Training using Bayesian Optimization
def optimize_xgboost(X_scaled, y):
    """
    Performs Bayesian hyperparameter optimization for XGBoost regressor.

    Args:
        X_scaled (np.ndarray): Scaled feature matrix.
        y (pd.Series): Scaled target variable.

    Returns:
        xgb.XGBRegressor: Best estimator found.
    """
    print("Starting model training...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method='hist',
        device='cuda',
        random_state=SEED
    )

    param_space = {
        'learning_rate': (0.01, 0.3, 'log-uniform'),
        'max_depth': (3, 10),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
        'lambda': (0.1, 10, 'log-uniform'),
        'alpha': (0.1, 10, 'log-uniform')
    }
    optimizer = BayesSearchCV(
        model,
        param_space,
        n_iter=30,
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    optimizer.fit(X_scaled, y)
    print("Model training completed.")
    return optimizer.best_estimator_


trained_xgb_model = optimize_xgboost(X_scaled, y)
trained_xgb_model.save_model("xgboost_model.json")
print("Model saved successfully.")
del X_scaled, y  # Free memory


# Process Validation Data
def process_validation_data(val_data, val_baseline):
    """
    Prepare validation data features based on baseline sample.

    Args:
        val_data (pd.DataFrame): Validation dataset excluding baseline.
        val_baseline (pd.Series): Baseline validation sample.

    Returns:
        tuple: Features DataFrame, actual concentrations Series,
               baseline RGB array, baseline concentration scalar.
    """
    baseline_rgb = val_baseline[['R', 'G', 'B']].values.astype(np.float32)
    baseline_concentration = max(val_baseline['Concentration'], 1e-6)

    X_val = (val_data[['R', 'G', 'B']] - baseline_rgb) / baseline_rgb * 100
    X_val['avg_pct_change'] = X_val.mean(axis=1)
    X_val['abs_avg_pct_change'] = X_val[['R', 'G', 'B']].abs().mean(axis=1)

    return X_val, val_data['Concentration'], baseline_rgb, baseline_concentration


X_val, y_val_actual, val_baseline_rgb, val_baseline_concentration = process_validation_data(val_data, val_baseline)
scaler = joblib.load("X_scaler.pkl")
X_val_scaled = scaler.transform(X_val)


# Validate Model
def validate_model(model, X_val_scaled, y_val_actual, val_baseline_concentration, threshold=0.1):
    """
    Evaluate model on validation data, compute success rate based on error threshold.

    Args:
        model (xgb.Booster): Trained XGBoost Booster model.
        X_val_scaled (np.ndarray): Scaled validation features.
        y_val_actual (pd.Series): True validation concentrations.
        val_baseline_concentration (float): Baseline concentration.
        threshold (float): Relative error threshold.

    Returns:
        float: Success rate as fraction of predictions within threshold.
    """
    dval = xgb.DMatrix(X_val_scaled)
    y_val_pred_scaled = model.predict(dval)

    # Load y scaler to inverse-transform predictions
    y_scaler_local = joblib.load("y_scaler.pkl")
    y_val_pred = y_scaler_local.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()

    # Add baseline concentration to predicted differences
    y_val_pred_final = val_baseline_concentration + y_val_pred

    # Calculate relative errors
    errors = np.abs((y_val_pred_final - y_val_actual) / y_val_actual)

    # Calculate success rate (percentage within threshold)
    success_rate = np.mean(errors <= threshold)
    return success_rate


trained_xgb_model = xgb.Booster()
trained_xgb_model.load_model("xgboost_model.json")
success_rate = validate_model(trained_xgb_model, X_val_scaled, y_val_actual, val_baseline_concentration)
print(f"Validation Success Rate: {success_rate * 100:.2f}%")


# Predict Unknown Samples
def predict_unknown_concentration_xgb(model, scaler, known_rgb, known_concentration, unknown_rgb):
    """
    Predict concentration for unknown RGB sample based on known RGB and concentration.

    Args:
        model (xgb.Booster): Trained XGBoost Booster model.
        scaler (StandardScaler): Feature scaler.
        known_rgb (list or np.array): Known RGB values.
        known_concentration (float): Known concentration.
        unknown_rgb (list or np.array): Unknown RGB values.

    Returns:
        float: Predicted concentration (non-negative).
    """
    known_rgb = np.array(known_rgb, dtype=np.float32)
    unknown_rgb = np.array(unknown_rgb, dtype=np.float32)

    # Avoid division by zero by replacing 0 with 1
    known_rgb = np.where(known_rgb == 0, 1, known_rgb)
    known_concentration = max(known_concentration, 1e-6)

    # Calculate percent change features
    pct_change = (unknown_rgb - known_rgb) / known_rgb * 100
    features = pd.DataFrame([list(pct_change) + [pct_change.mean(), np.abs(pct_change).mean()]],
                            columns=['R', 'G', 'B', 'avg_pct_change', 'abs_avg_pct_change'])

    # Scale features and prepare for prediction
    features_scaled = scaler.transform(features)
    dfeatures = xgb.DMatrix(features_scaled)

    concentration_difference_pred = model.predict(dfeatures)[0]

    # Load y scaler to inverse transform predicted difference
    y_scaler_local = joblib.load("y_scaler.pkl")
    concentration_difference_pred = y_scaler_local.inverse_transform([[concentration_difference_pred]])[0][0]

    # Calculate final predicted concentration, ensuring non-negative result
    return max(0, known_concentration + concentration_difference_pred)


# Example Prediction
scaler = joblib.load("X_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

known_rgb = [184.252, 115.941, 34.988]
known_concentration = 0.0000001
unknown_rgb = [166.319, 86.068, 21.406]

predicted_concentration = predict_unknown_concentration_xgb(trained_xgb_model, scaler, known_rgb, known_concentration,
                                                            unknown_rgb)
print(f"Predicted Concentration: {predicted_concentration:.6f} nM")
