import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# -----------------------------------------
# Monte Carlo Estimator for 1-F for a given φ
# -----------------------------------------
def one_minus_F_MC(theta, phi, lambda1, lambda2, N=2000):
    """
    Computes 1 - F_Y(t) via Monte Carlo, where
      Y = cos(φ)*X1 + sin(φ)*X2,
      t = θ · (cos(φ), sin(φ)),
    and X1 ~ Exp(λ₁), X2 ~ Exp(λ₂).
    """
    a = np.cos(phi)
    b = np.sin(phi)
    t = theta[0] * a + theta[1] * b  # projection of theta
    X1 = np.random.exponential(scale=1 / lambda1, size=N)
    X2 = np.random.exponential(scale=1 / lambda2, size=N)
    Y = a * X1 + b * X2
    F_est = np.mean(Y <= t)
    return 1 - F_est


# -----------------------------------------
# Dichotomy (Bisection) Search for Optimal φ
# -----------------------------------------
def find_optimal_phi_dichotomy(
    theta, lambda1, lambda2, phi_min=0, phi_max=2 * np.pi, tol=1e-4, max_iter=50, N=2000
):
    a_phi = phi_min
    b_phi = phi_max
    delta = tol / 2.0
    for _ in range(max_iter):
        mid = (a_phi + b_phi) / 2.0
        phi1 = mid - delta
        phi2 = mid + delta
        f1 = one_minus_F_MC(theta, phi1, lambda1, lambda2, N)
        f2 = one_minus_F_MC(theta, phi2, lambda1, lambda2, N)
        if f1 < f2:
            b_phi = phi2
        else:
            a_phi = phi1
        if (b_phi - a_phi) < tol:
            break
    phi_opt = (a_phi + b_phi) / 2.0
    f_opt = one_minus_F_MC(theta, phi_opt, lambda1, lambda2, N)
    return phi_opt, f_opt


# -----------------------------------------
# Compute Tukey Depth for a Fixed θ
# -----------------------------------------
def compute_depth(theta, lambda1, lambda2, phi_points=50, N=2000):
    phi_grid = np.linspace(0, 2 * np.pi, phi_points)
    values = np.array(
        [one_minus_F_MC(theta, phi, lambda1, lambda2, N) for phi in phi_grid]
    )
    depth = np.min(values)
    opt_phi = phi_grid[np.argmin(values)]
    return depth, opt_phi


# -----------------------------------------
# Compute Contour Data with a Progress Bar
# -----------------------------------------
def compute_contour(
    lambda1, lambda2, theta1_range, theta2_range, grid_points=30, phi_points=30, N=1000
):
    theta1_vals = np.linspace(theta1_range[0], theta1_range[1], grid_points)
    theta2_vals = np.linspace(theta2_range[0], theta2_range[1], grid_points)
    Depth = np.zeros((grid_points, grid_points))

    progress_bar = st.progress(0)
    total = grid_points
    for i, th1 in enumerate(theta1_vals):
        for j, th2 in enumerate(theta2_vals):
            theta = [th1, th2]
            d, _ = compute_depth(theta, lambda1, lambda2, phi_points, N)
            Depth[j, i] = d  # j-index for y-axis, i-index for x-axis
        progress_bar.progress((i + 1) / total)
    return theta1_vals, theta2_vals, Depth


# =========================================
# Streamlit App Layout
# =========================================

st.title("Tukey Depth Interactive Visualization (Localhost)")

# Sidebar inputs
st.sidebar.header("Input Parameters")
lambda1 = st.sidebar.number_input("λ₁", value=2.0, step=0.1)
lambda2 = st.sidebar.number_input("λ₂", value=1.5, step=0.1)

st.sidebar.header("Contour Plot Settings")
theta1_min = st.sidebar.number_input("θ₁ min", value=0.0)
theta1_max = st.sidebar.number_input("θ₁ max", value=3.0)
theta2_min = st.sidebar.number_input("θ₂ min", value=0.0)
theta2_max = st.sidebar.number_input("θ₂ max", value=3.0)
grid_points = st.sidebar.slider("Grid resolution", min_value=10, max_value=50, value=20)

# ---- Contour Plot Section (Graph 1) ----
st.header("Contour Plot of Tukey Depth")
with st.spinner("Computing contour plot..."):
    theta1_vals, theta2_vals, Depth = compute_contour(
        lambda1,
        lambda2,
        [theta1_min, theta1_max],
        [theta2_min, theta2_max],
        grid_points=grid_points,
    )
fig1, ax1 = plt.subplots()
T1, T2 = np.meshgrid(theta1_vals, theta2_vals)
cp = ax1.contourf(T1, T2, Depth, cmap="viridis")
fig1.colorbar(cp, ax=ax1, label="Tukey Depth (min 1-F)")
ax1.set_xlabel("θ₁")
ax1.set_ylabel("θ₂")
ax1.set_title("Contour Plot of Tukey Depth")
st.pyplot(fig1)

# ---- Interactive 1-F Plot Section (Graph 2) ----
st.header("1 - F(θ) vs φ for a Selected θ")
theta_x = st.slider("θ₁", min_value=0.0, max_value=3.0, value=1.0)
theta_y = st.slider("θ₂", min_value=0.0, max_value=3.0, value=1.0)
theta = [theta_x, theta_y]

phi_points = 100
phi_grid = np.linspace(0, 2 * np.pi, phi_points)
values = [one_minus_F_MC(theta, phi, lambda1, lambda2, N=2000) for phi in phi_grid]
depth_val, opt_phi = compute_depth(
    theta, lambda1, lambda2, phi_points=phi_points, N=2000
)
fig2, ax2 = plt.subplots()
ax2.plot(phi_grid, values, label="1 - F(θ)")
ax2.axvline(opt_phi, color="r", linestyle="--", label=f"Optimal φ ≈ {opt_phi:.2f}")
ax2.set_xlabel("φ (radians)")
ax2.set_ylabel("1 - F(θ)")
ax2.set_title("Variation of 1 - F(θ) with φ")
ax2.legend()
st.pyplot(fig2)

# ---- Half-Plane Visualization (Graph 3) ----
st.header("Half-Plane Visualization for the Optimal Projection")
st.write("""
For the selected point θ and its optimal projection direction φ, the following plot shows:
- The line where the projection equals t = θ · (cos φ, sin φ).
- The half-plane { x ∈ ℝ² : ⟨x, u⟩ ≥ t } is highlighted in light blue.
""")

# Use the optimal φ computed from Graph 2.
u = np.array([np.cos(opt_phi), np.sin(opt_phi)])
t_val = theta[0] * u[0] + theta[1] * u[1]

# Define the plotting region.
x_min, x_max = 0, 3
y_min, y_max = 0, 3
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
proj = xx * u[0] + yy * u[1]
mask = proj >= t_val

fig3, ax3 = plt.subplots()
# Convert mask to integer: values 0 and 1.
mask_int = mask.astype(int)
# Fill two regions: when mask_int==0 (white) and when mask_int==1 (light blue).
ax3.contourf(
    xx, yy, mask_int, levels=[-0.1, 0.5, 1.1], colors=["white", "lightblue"], alpha=0.5
)
# Draw the separating line where proj == t_val.
CS = ax3.contour(xx, yy, proj, levels=[t_val], colors="red", linewidths=2)
ax3.clabel(CS, inline=True, fontsize=10)
ax3.plot(theta[0], theta[1], "ko", label="θ")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_title(f"Half-Plane for Optimal Projection (φ ≈ {opt_phi:.2f} rad)")
ax3.legend()
st.pyplot(fig3)
