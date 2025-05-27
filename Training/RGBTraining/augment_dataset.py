import pandas as pd
import numpy as np


def estimate_noise_std(df: pd.DataFrame) -> float:
    """
    Estimates the standard deviation of noise in the dataset by computing
    the standard deviation of differences between consecutive RGB values.

    Args:
        df (pd.DataFrame): Input dataset with 'R', 'G', 'B' columns.

    Returns:
        float: Estimated standard deviation of noise.
    """
    # Extract RGB values as a NumPy array for numerical operations
    rgb_values = df[['R', 'G', 'B']].values

    # Compute differences between consecutive RGB rows to capture noise variation
    diffs = np.diff(rgb_values, axis=0)

    # Calculate and return the overall standard deviation of these differences as noise estimate
    return np.std(diffs)


def augment_dataset(
        df: pd.DataFrame,
        num_points: int = 500
) -> pd.DataFrame:
    """
    Augments a dataset by interpolating and perturbing values using estimated noise.

    Args:
        df (pd.DataFrame): The input dataset containing 'Concentration', 'R', 'G', 'B'.
        num_points (int): Number of interpolation points between each pair of rows.

    Returns:
        pd.DataFrame: Augmented dataset with interpolated and perturbed values.
    """
    # Validate minimum number of rows for interpolation
    if df.shape[0] < 2:
        raise ValueError("Dataset must have at least two rows.")

    # Validate required columns presence
    required_cols = {'Concentration', 'R', 'G', 'B'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    # Estimate noise standard deviation from RGB differences
    noise_std = estimate_noise_std(df)
    print(f"Estimated noise standard deviation: {noise_std:.4f}")

    augmented_data = []  # List to accumulate augmented rows

    concentrations = df['Concentration'].values
    rgb_values = df[['R', 'G', 'B']].values

    # Loop through each pair of consecutive points for interpolation
    for i in range(len(concentrations) - 1):
        # Generate evenly spaced weights for interpolation (exclude endpoint to avoid duplicate points)
        weights = np.linspace(0, 1, num_points, endpoint=False)

        # Linear interpolation for concentrations between two points
        interp_concentrations = (1 - weights) * concentrations[i] + weights * concentrations[i + 1]

        # Linear interpolation for RGB channels between two points
        interp_rgb = (1 - weights[:, None]) * rgb_values[i] + weights[:, None] * rgb_values[i + 1]

        # Add Gaussian noise with estimated noise std dev to RGB values
        perturbed_rgb = interp_rgb + np.random.normal(scale=noise_std, size=interp_rgb.shape)

        # Clip RGB values to valid range [0, 255] assuming 8-bit colors
        perturbed_rgb = np.clip(perturbed_rgb, 0, 255)

        # Append each interpolated and perturbed sample as a row to augmented data
        for conc, rgb in zip(interp_concentrations, perturbed_rgb):
            augmented_data.append([conc, *rgb])

    # Return a DataFrame with appropriate column names
    return pd.DataFrame(augmented_data, columns=['Concentration', 'R', 'G', 'B'])


if __name__ == "__main__":
    # Load dataset from CSV and drop rows with any missing values
    data = pd.read_csv('Input.csv')
    data = data.dropna(how='any')

    # Generate augmented dataset with interpolation and noise perturbation
    augmented_data = augment_dataset(data, num_points=500)

    # Save the augmented dataset to CSV without the index column
    augmented_data.to_csv('output.csv', index=False)
