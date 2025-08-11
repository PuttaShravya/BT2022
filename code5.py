import numpy as np
from scipy.stats import t
from numpy.linalg import pinv, cond, inv, norm
import matplotlib.pyplot as plt
from tabulate import tabulate

def to_subscript(n):
    subscript_digits = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(n).translate(subscript_digits)

alpha = 0.0456565

tol = 1e-05 # tol to exit
itr = 4 # Gauss-Newton iterations to reach tol

## Submit these 5 answers

final_JTJ_index = (3,3) # CHANGE (index_1, index_2)
final_JTJ_inverse_index = (1,1) # CHANGE (index_1, index_2)
Cov_Beta_index = (1,3) # CHANGE (index_1, index_2)

# kth parameter: x
# kth parameter low/high at alpha: "LOW/HIGH"
# p-value parameter number = y

ci_param_x, p_value_param_y = (["U", 1], 3) # CHANGE
## CHANGE x and y in (["L/U", x], y) as per your data by comparing with given comments


data = [
[0, 0.00785582],
[0.025, 0.0108258],
[0.05, 0.007674],
[0.075, 0.0131186],
[0.1, 0.00770421],
[0.125, 0.00331499],
[0.15, 0.00163713],
[0.175, -0.00184932],
[0.2, -0.00574485],
[0.225, -0.00339752],
[0.25, -0.029413],
[0.275, -0.0176918],
[0.3, -0.0258805],
[0.325, -0.0299822],
[0.35, -0.0505325],
[0.375, -0.0598151],
[0.4, -0.06959],
[0.425, -0.0821377],
[0.45, -0.0864049],
[0.475, -0.0972821],
[0.5, -0.113271],
[0.525, -0.131802],
[0.55, -0.143181],
[0.573, -0.161545],
[0.6, -0.177852],
[0.625, -0.181547],
[0.65, -0.210765],
[0.675, -0.221731],
[0.7, -0.247172],
[0.725, -0.257016],
[0.75, -0.27625],
[0.775, -0.309827],
[0.8, -0.330728],
[0.825, -0.339997],
[0.85, -0.374163],
[0.875, -0.401543],
[0.9, -0.41518],
[0.925, -0.450336],
[0.95, -0.469287],
[0.975, -0.503588],
[1, -0.528046],
[1.025, -0.549151],
[1.05, -0.579201],
[1.075, -0.606176],
[1.1, -0.626163],
[1.125, -0.654696],
[1.15, -0.692488],
[1.175, -0.700428],
[1.2, -0.717625],
[1.225, -0.733888],
[1.25, -0.75941],
[1.275, -0.778529],
[1.3, -0.780636],
[1.325, -0.799004],
[1.35, -0.807352],
[1.375, -0.815948],
[1.4, -0.822299],
[1.425, -0.817198],
[1.45, -0.817448],
[1.475, -0.81365],
[1.5, -0.820662],
[1.525, -0.817726],
[1.55, -0.803904],
[1.575, -0.790341],
[1.6, -0.789827],
[1.625, -0.780659],
[1.65, -0.773468],
[1.675, -0.763091],
[1.7, -0.749466],
[1.725, -0.749952],
[1.75, -0.735625],
[1.775, -0.734307],
[1.8, -0.71983],
[1.825, -0.711234],
[1.85, -0.693604],
[1.875, -0.684495],
[1.9, -0.686555],
[1.925, -0.675438],
[1.95, -0.657102],
[1.975, -0.664453],
[2, -0.654598],
[2.025, -0.641991],
[2.05, -0.631596],
[2.075, -0.635967],
[2.1, -0.619671],
[2.125, -0.619001],
[2.15, -0.604577],
[2.175, -0.593636],
[2.2, -0.593256],
[2.225, -0.581909],
[2.25, -0.576616],
[2.275, -0.568498],
[2.3, -0.569652],
[2.325, -0.573016],
[2.35, -0.562624],
[2.375, -0.550913],
[2.4, -0.543737],
[2.425, -0.534413],
[2.45, -0.520823],
[2.475, -0.523962],
[2.5, -0.515181],
[2.525, -0.516567],
[2.55, -0.508447],
[2.575, -0.516553],
[2.6, -0.495386],
[2.625, -0.493159],
[2.65, -0.499966],
[2.675, -0.489196],
[2.7, -0.484829],
[2.725, -0.475727],
[2.75, -0.46852],
[2.775, -0.472564],
[2.8, -0.46923],
[2.825, -0.470532],
[2.85, -0.457775],
[2.875, -0.449136],
[2.9, -0.449545],
[2.925, -0.455905],
[2.95, -0.44256],
[2.975, -0.43996],
[3, -0.43216],
[3.025, -0.436681],
[3.05, -0.435353],
[3.075, -0.426767],
[3.1, -0.424219],
[3.125, -0.427554],
[3.15, -0.417956],
[3.175, -0.418038],
[3.2, -0.407668],
[3.225, -0.409533],
[3.25, -0.403594],
[3.275, -0.391789],
[3.3, -0.395486],
[3.325, -0.389301],
[3.35, -0.402182],
[3.375, -0.387326],
[3.4, -0.371561],
[3.425, -0.386998],
[3.45, -0.382357],
[3.475, -0.375767],
[3.5, -0.364537],
[3.525, -0.375429],
[3.55, -0.364991],
[3.575, -0.367456],
[3.6, -0.371377],
[3.625, -0.369182],
[3.65, -0.36434],
[3.675, -0.363583],
[3.7, -0.342574],
[3.725, -0.345151],
[3.75, -0.347359],
[3.775, -0.342492],
[3.8, -0.343962],
[3.825, -0.334942],
[3.85, -0.339571],
[3.875, -0.348958],
[3.9, -0.328916],
[3.925, -0.321435],
[3.95, -0.331716],
[3.975, -0.332404],
[4, -0.322907]
]

np_data = np.array(data)
JTJ_idx = tuple(i - 1 if i<5 else i-2 for i in final_JTJ_index)
JTJ_inverse_idx = tuple(i - 1 if i<5 else i-2 for i in final_JTJ_inverse_index)
Covariance_beta_idx = tuple(i - 1 if i<5 else i-2 for i in Cov_Beta_index)

# Extract variables
X = np_data[:, 0]
Y = np_data[:, 1]

# Initial Setup
y0 = Y[0]
b = np.array([1.0, 1.0, 1.0, 1.0])
max_iter = itr

N,p = len(Y),len(b)
DOF = N - p

# ODE Model
def ode_model(x, y, b):
    b0, b1, b2, b3 = b
    return b0 * x**2 * y**4 + b1 * y**3 + b2 * x**3 * y**2 + b3 * y - x

# Euler Solver
def ode_solve(b, x_vals, y_init):
    try:
        y_pred = np.zeros_like(x_vals)
        y = y_init
        y_pred[0] = y
        for i in range(1, len(x_vals)):
            h = x_vals[i] - x_vals[i - 1]
            dydx = ode_model(x_vals[i - 1], y, b)
            y += h * dydx
            y_pred[i] = y
        return y_pred, True
    except Exception as e:
        print(f"ODE solver error: {e}")
        return None, False

# Jacobian Calculation
def calc_jacobian(b, x_vals, y_init, delta=1e-5):
    J = np.zeros((len(x_vals), len(b)))
    y_base, success = ode_solve(b, x_vals, y_init)
    if not success:
        print("Base ODE solve failed")
        return None, False

    for j in range(len(b)):
        b_pert = b.copy()
        step = delta * abs(b[j]) if b[j] != 0 else delta
        b_pert[j] += step
        y_pert, _ = ode_solve(b_pert, x_vals, y_init)
        J[:, j] = (y_pert - y_base) / step

    return J, True

# Iterative Optimization
for iter in range(1, max_iter + 1):
    y_pred, success = ode_solve(b, X, y0)
    if not success:
        print(f"ODE solver failed at iteration {iter}")
        break

    res = Y - y_pred
    SSR = np.dot(res, res)

    J, success = calc_jacobian(b, X, y0)
    if not success:
        print(f"Jacobian failed at iteration {iter}")
        break

    JTJ = J.T @ J
    if cond(JTJ) > 1 / np.finfo(float).eps:
        print(f"Matrix ill-conditioned at iteration {iter}, using pseudo-inverse.")
        del_b = pinv(JTJ) @ (J.T @ res)
    else:
        del_b = np.linalg.solve(JTJ, J.T @ res)

    b_new = b + del_b
    norm_del_b = norm(del_b)

    # Uncomment to check iteration results
    # print(f"{iter:4d} | SSR = {SSR:.4e} | norm(del_b) = {norm_del_b:.4e} | b = {b}")

    b = b_new

    if iter == max_iter or norm_del_b < tol:
        y_pred, _ = ode_solve(b, X, y0)
        res = Y - y_pred
        SSR = np.dot(res, res)
        J, _ = calc_jacobian(b, X, y0)
        JTJ = J.T @ J
        JTJ_inv = inv(JTJ)
        break

# Final Reporting
if norm_del_b < tol or iter == max_iter:

    s2 = SSR / DOF
    cov_matrix = s2 * JTJ_inv

    se = np.sqrt(np.diag(cov_matrix))
    t_crit = t.ppf(1 - alpha / 2, DOF)

    ci_low = b - np.abs(t_crit) * se
    ci_high = b + np.abs(t_crit) * se
    t_stat = b / se
    p_val = 2 * (t.sf(np.abs(t_stat), DOF))


    if ci_param_x[0] == "U":
      CI_param_x = ci_high[ci_param_x[1]-1]
    elif ci_param_x[0] == "L":
      CI_param_x = ci_low[ci_param_x[1]-1]

    p_Val_param_y = p_val[p_value_param_y-1]

    # Print results
    print("\n******* ANSWERS TO SUBMIT *******\n")
    print(f"Element {final_JTJ_index} of JᵀJ : {JTJ[JTJ_idx]:.8f}")
    print(f"Element {final_JTJ_inverse_index} of (JᵀJ)⁻¹ : {JTJ_inv[JTJ_inverse_idx]:.8f}")
    print(f"Element {Cov_Beta_index} of Cov(β) : {cov_matrix[Covariance_beta_idx]:6e}")
    print("Lower" if ci_param_x[0] == "L" else "Upper", f"bound for b{to_subscript(ci_param_x[1]-1)} at given alpha: {CI_param_x:.8f}")
    print(f"p-value for b{to_subscript(p_value_param_y-1)}: {p_Val_param_y:.6e}\n\n")

    
