import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)

# ==============================================================================
# Dataset Definition
# ==============================================================================
X = np.array([0.5, 2.3, 2.9])
y = np.array([1.4, 1.9, 3.2])


# ==============================================================================
# TASK 1: Ordinary Least Squares (OLS) Implementation
# ==============================================================================
def ols_linear_regression(X, y):
    """
    Compute linear regression parameters analytically using OLS closed-form formulas.

    Parameters:
        X (np.ndarray): Input feature vector (Height)
        y (np.ndarray): Target vector (Weight)

    Returns:
        m (float): Optimal slope
        c (float): Optimal intercept
    """
    # Compute mean of X and mean of y
    x_mean = np.mean(X)
    y_mean = np.mean(y)

    # Compute numerator and denominator for slope 'm'
    numerator = np.sum((X - x_mean) * (y - y_mean))
    denominator = np.sum((X - x_mean) ** 2)
    m = numerator / denominator

    # Compute intercept 'c'
    c = y_mean - m * x_mean

    return float(m), float(c)


# ==============================================================================
# TASK 2: Batch Gradient Descent Implementation
# ==============================================================================
def gradient_descent(X, y, alpha=0.05, epochs=1000):
    """
    Optimize linear regression parameters using batch gradient descent.

    Parameters:
        X (np.ndarray): Input feature vector
        y (np.ndarray): Target vector
        alpha (float): Learning rate
        epochs (int): Total training iterations

    Returns:
        m (float): Learned slope
        c (float): Learned intercept
        loss_history (list): MSE loss recorded per epoch
    """
    m = 0.0
    c = 0.0
    n = len(X)
    loss_history = []

    for epoch in range(epochs):
        # Compute linear predictions (y_pred = m * X + c)
        y_pred = m * X + c

        # Compute Mean Squared Error (MSE) loss and store in loss_history
        loss = np.mean((y_pred - y) ** 2)
        loss_history.append(float(loss))

        # Compute partial derivatives dm and dc
        dm = (2.0 / n) * np.sum((y_pred - y) * X)
        dc = (2.0 / n) * np.sum(y_pred - y)

        # Update parameters m and c using gradients and learning rate alpha
        m = m - alpha * dm
        c = c - alpha * dc

    return float(m), float(c), loss_history


# ==============================================================================
# TASK 3: Model Evaluation Metrics (R2, t-statistic, and p-value)
# ==============================================================================
def compute_evaluation_metrics(X, y, y_pred, m):
    """
    Calculate R-squared and statistical significance (p-value) of the slope parameter.

    Parameters:
        X (np.ndarray): Input feature vector
        y (np.ndarray): Target ground truth vector
        y_pred (np.ndarray): Predicted values from model
        m (float): Slope parameter

    Returns:
        r2 (float): Coefficient of determination
        t_stat (float): Student's t-statistic for slope
        p_val (float): Two-tailed p-value
    """
    n = len(X)
    df = n - 2

    # Calculate SS_res (residual sum of squares) and SS_tot (total sum of squares)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    # Calculate R-squared (R2)
    r2 = 1.0 - (ss_res / ss_tot)

    # Calculate Standard Error of the slope SE(m)
    # residual_variance = ss_res / df
    residual_variance = ss_res / df if df > 0 else 0.0
    x_mean = np.mean(X)
    denom_var = np.sum((X - x_mean) ** 2)
    se_m = np.sqrt(residual_variance / denom_var)

    # Compute t-statistic (t = m / se_m)
    t_stat = m / se_m if se_m > 0 else np.nan

    # Compute two-tailed p-value using stats.t.sf or stats.t.cdf
    p_val = 2.0 * stats.t.sf(np.abs(t_stat), df=df) if df > 0 else np.nan

    return float(r2), float(t_stat), float(p_val)


# ==============================================================================
# TASK 4: Model Execution & Evaluation Comparison
# ==============================================================================
def main():
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("=" * 60)
    log("LAB 05: ORDINARY LEAST SQUARES (OLS) & GRADIENT DESCENT")
    log("=" * 60)
    log(f"Dataset X (Height): {X}")
    log(f"Dataset y (Weight): {y}")
    log(f"Sample size n = {len(X)}, Degrees of freedom df = {len(X) - 2}")

    # --- Execute OLS ---
    m_ols, c_ols = ols_linear_regression(X, y)
    y_pred_ols = m_ols * X + c_ols
    r2_ols, t_ols, p_ols = compute_evaluation_metrics(X, y, y_pred_ols, m_ols)

    log("\n==================== OLS RESULTS ====================")
    log(f"Equation       : y = {m_ols:.4f}x + {c_ols:.4f}")
    log(f"Slope (m)      : {m_ols:.6f}")
    log(f"Intercept (c)  : {c_ols:.6f}")
    log(f"R-squared      : {r2_ols:.4f}")
    log(f"t-statistic    : {t_ols:.4f}")
    log(f"p-value        : {p_ols:.4f}")

    # --- Execute Gradient Descent ---
    alpha = 0.05
    epochs = 1500
    m_gd, c_gd, losses = gradient_descent(X, y, alpha=alpha, epochs=epochs)
    y_pred_gd = m_gd * X + c_gd
    r2_gd, t_gd, p_gd = compute_evaluation_metrics(X, y, y_pred_gd, m_gd)

    log("\n============== GRADIENT DESCENT RESULTS =============")
    log(f"Equation       : y = {m_gd:.4f}x + {c_gd:.4f}")
    log(f"Slope (m)      : {m_gd:.6f}")
    log(f"Intercept (c)  : {c_gd:.6f}")
    log(f"R-squared      : {r2_gd:.4f}")
    log(f"t-statistic    : {t_gd:.4f}")
    log(f"p-value        : {p_gd:.4f}")
    log(f"Final Loss     : {losses[-1]:.6f}")

    # --- Comparison Table ---
    log("\n================ SIDE-BY-SIDE COMPARISON ================")
    log(f"{'Metric / Parameter':<22} | {'Analytical (OLS)':<16} | {'Iterative (GD)':<16} | {'Difference':<12}")
    log("-" * 72)
    log(f"{'Slope (m)':<22} | {m_ols:<16.6f} | {m_gd:<16.6f} | {abs(m_ols - m_gd):<12.6f}")
    log(f"{'Intercept (c)':<22} | {c_ols:<16.6f} | {c_gd:<16.6f} | {abs(c_ols - c_gd):<12.6f}")
    log(f"{'R-squared':<22} | {r2_ols:<16.6f} | {r2_gd:<16.6f} | {abs(r2_ols - r2_gd):<12.6f}")
    log(f"{'t-statistic':<22} | {t_ols:<16.6f} | {t_gd:<16.6f} | {abs(t_ols - t_gd):<12.6f}")
    log(f"{'p-value':<22} | {p_ols:<16.6f} | {p_gd:<16.6f} | {abs(p_ols - p_gd):<12.6f}")

    # --- TASK 5: Plotting & Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Regression Fit Comparison
    x_vals = np.linspace(0, 3.5, 100)
    axes[0].scatter(X, y, color="#1f77b4", s=100, label="Actual Data Points", zorder=5)
    axes[0].plot(x_vals, m_ols * x_vals + c_ols, color="#2ca02c", linestyle="-", linewidth=2.5, label=f"OLS: y = {m_ols:.2f}x + {c_ols:.2f}")
    axes[0].plot(x_vals, m_gd * x_vals + c_gd, color="#d62728", linestyle="--", linewidth=2, label=f"GD: y = {m_gd:.2f}x + {c_gd:.2f}")
    axes[0].set_title("OLS vs Gradient Descent Regression Line", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Height (X)", fontsize=11)
    axes[0].set_ylabel("Weight (y)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Subplot 2: Loss Convergence Curve
    axes[1].plot(range(1, epochs + 1), losses, color="#9467bd", linewidth=2)
    axes[1].set_title("MSE Loss Convergence over Epochs", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Mean Squared Error (Loss)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig_path = "figures/regression_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    log(f"\nSaved regression plot to {fig_path}")

    # --- Learning Rate Sensitivity Analysis ---
    log("\n========== LEARNING RATE SENSITIVITY EXPERIMENTS ==========")
    rates = [1.0, 0.05, 0.0001]
    plt.figure(figsize=(10, 5))

    for lr in rates:
        m_tmp, c_tmp, l_tmp = gradient_descent(X, y, alpha=lr, epochs=100)
        log(f"Alpha = {lr:<6}: Final m={m_tmp:10.4f}, c={c_tmp:10.4f}, Final Loss={l_tmp[-1]:.6e}")
        # Plot up to 50 epochs for sensitivity visualization
        display_losses = [min(l, 100.0) for l in l_tmp[:50]]
        plt.plot(range(1, len(display_losses) + 1), display_losses, label=f"alpha = {lr}")

    plt.title("Learning Rate Sensitivity Comparison (First 50 Epochs)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("MSE Loss (Capped at 100 for visibility)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    lr_fig_path = "figures/lr_sensitivity.png"
    plt.savefig(lr_fig_path, dpi=300)
    plt.close()
    log(f"Saved learning rate sensitivity plot to {lr_fig_path}")

    # Write output_log.txt
    with open("output_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log("Saved output log to output_log.txt")


if __name__ == "__main__":
    main()
