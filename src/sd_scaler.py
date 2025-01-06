import numpy as np
from scipy.optimize import minimize


def rescale_with_bounds(numbers, target_sd=7.3229, min_bound=0.0):
    """
    Rescale numbers to target SD while maintaining relative ordering for small values.
    """
    numbers = np.array(numbers)
    original_mean = np.mean(numbers)

    # Split into small (<1) and large values but scale both
    small_mask = (numbers < 1)
    large_mask = ~small_mask

    # Get reference ranges for maintaining proportions of small values
    small_values = numbers[small_mask]
    if len(small_values) > 0:
        small_min = np.min(small_values)
        small_max = np.max(small_values)

    def objective(params):
        # params[0] is scale factor for large values
        # params[1] is scale factor for small values
        # params[2] is shift for small values

        result = numbers.copy()

        # Scale large values
        if np.any(large_mask):
            large_centered = numbers[large_mask] - np.mean(numbers[large_mask])
            result[large_mask] = large_centered * params[0] + np.mean(numbers[large_mask])

        # Scale small values while preserving order
        if np.any(small_mask):
            small_scaled = (numbers[small_mask] - small_min) / (small_max - small_min)
            small_scaled = small_scaled * params[1] + params[2]
            result[small_mask] = small_scaled

        current_sd = np.std(result)
        current_mean = np.mean(result)

        # Penalties
        sd_penalty = (current_sd - target_sd) ** 2 * 3000
        mean_penalty = (current_mean - original_mean) ** 2 * 800
        min_penalty = np.sum(np.maximum(0, min_bound - result) ** 2) * 1000

        # Penalty for small values order violation
        order_penalty = 0
        if np.any(small_mask):
            small_diffs = np.diff(result[small_mask])
            order_penalty = np.sum(np.maximum(0, -small_diffs)) * 2000

        return sd_penalty + mean_penalty + min_penalty + order_penalty

    # Initial guess
    initial_guess = [
        target_sd / np.std(numbers),  # large values scale
        0.5,  # small values scale
        min_bound  # small values shift
    ]

    # Optimize
    result = minimize(objective, initial_guess, method='Nelder-Mead',
                      options={'maxiter': 2000})

    # Apply final transformation
    final_result = numbers.copy()

    # Apply to large values
    if np.any(large_mask):
        large_centered = numbers[large_mask] - np.mean(numbers[large_mask])
        final_result[large_mask] = large_centered * result.x[0] + np.mean(numbers[large_mask])

    # Apply to small values
    if np.any(small_mask):
        small_scaled = (numbers[small_mask] - small_min) / (small_max - small_min)
        small_scaled = small_scaled * result.x[1] + result.x[2]
        final_result[small_mask] = small_scaled

    # Ensure minimum bound
    final_result = np.maximum(final_result, min_bound)

    return final_result


# Test the function
import pandas as pd

# Read the CSV
df = pd.read_csv('sd_table.csv')
original_values = df['Predicted Ownership'].values

# Scale the values
scaled = rescale_with_bounds(original_values)

# Print statistics
print(f"Original Statistics:")
print(f"Mean: {np.mean(original_values):.4f}")
print(f"SD: {np.std(original_values):.4f}")
print(f"Min: {np.min(original_values):.4f}")
print(f"Max: {np.max(original_values):.4f}")
print(f"Number of values below 1: {np.sum(original_values < 1)}")

print(f"\nScaled Statistics:")
print(f"Mean: {np.mean(scaled):.4f}")
print(f"SD: {np.std(scaled):.4f}")
print(f"Min: {np.min(scaled):.4f}")
print(f"Max: {np.max(scaled):.4f}")
print(f"Number of values below 1: {np.sum(scaled < 1)}")

# Show some small value comparisons
small_orig = original_values[original_values < 1]
small_scaled = scaled[original_values < 1]
print("\nSample of small value scaling (first 5):")
print(pd.DataFrame({
    'Original': small_orig[:5],
    'Scaled': small_scaled[:5]
}))

# Save results
df_scaled = df.copy()
df_scaled['Scaled Ownership'] = scaled


# Save results
df_scaled.to_csv('scaled_output.csv', index=False)