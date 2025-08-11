import numpy as np
import statsmodels.api as sm
from scipy import stats
import re
from collections import Counter
from functools import reduce


def to_subscript(n):
    subscript_digits = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(n).translate(subscript_digits)
  
alpha = 0.0419121

XTX_index = (3,4) 
XTX_inverse_index = (4,5)
Covariance_beta_index = (2,4)  
ci_beta_x, p_value_beta_y = (["U", 1], 3)  


# Model string
Y = "beta_0 + beta_1 X4X3X3X2 + beta_2 X2X3X3X1 + beta_3 X2X3X4X4 + beta_4 X4X3X1X3" # Keep whitespaces between beta_i and X2.....X4

## X1, X2, X3, X4, Y

data = [[0.00131788, 0.00105795, 0.00312036, -0.023515, 2.55615],
[0.0808791, 0.0978253, 0.0802341, 0.0931541, 2.52994] ,
[0.210344, 0.185975, 0.204336, 0.200691, 2.50645] ,
[0.306239, 0.305734, 0.308365, 0.281779, 2.45848] ,
[0.400222, 0.408365, 0.408724, 0.395987, 2.41763],
[0.504029, 0.505039, 0.50518, 0.490649, 2.22783] ,
[0.602741, 0.608465, 0.589597, 0.606042, 1.9344] ,
[0.7107, 0.694768, 0.701067, 0.702244, 1.5093] ,
[0.824149, 0.793146, 0.800238, 0.788402, 0.815695] ,
[0.899266, 0.905987, 0.899099, 0.88138, -0.20983] ,
[0.99421, 0.988418, 0.986091, 1.00678, -1.84229] ,
[1.10633, 1.09873, 1.07834, 1.09413, -3.74118] ,
[1.20353, 1.201, 1.19319, 1.18598, -6.26194] ,
[1.3026, 1.27619, 1.30516, 1.29438, -9.61809] ,
[1.39844, 1.40025, 1.40449, 1.39973, -14.2361] ,
[1.49769, 1.50213, 1.49924, 1.48106, -18.922] ,
[1.59759, 1.58926, 1.61003, 1.57819, -25.0319] ,
[1.69047, 1.69591, 1.70572, 1.6833, -32.985] ,
[1.80695, 1.80152, 1.79256, 1.81083, -43.8965],
[1.92068, 1.88072, 1.89247, 1.88467, -52.7388] ,
[1.99419, 2.00196, 1.98412, 2.00415, -67.2894] ,
[2.09787, 2.10652, 2.09252, 2.09925, -82.2704] ,
[2.20669, 2.19898, 2.20514, 2.1897, -98.6456] ,
[2.28861, 2.29532, 2.31156, 2.29098, -118.351] ,
[2.41774, 2.40471, 2.39515, 2.39486, -141.712] ,
[2.50347, 2.51225, 2.51186, 2.50167, -169.286] ,
[2.59518, 2.59839, 2.60921, 2.60153, -197.164] ,
[2.71071, 2.71626, 2.69141, 2.70813, -231.943] ,
[2.79003, 2.78667, 2.798, 2.79423, -262.665] ,
[2.89169, 2.90231, 2.92039, 2.89274, -305.202]]

np_data = np.array(data)
XTX_idx = tuple(i - 1 for i in XTX_index)
XTX_inverse_idx = tuple(i - 1 for i in XTX_inverse_index)
Covariance_beta_idx = tuple(i - 1 for i in Covariance_beta_index)

# Extract variables
x1 = np_data[:, 0]
x2 = np_data[:, 1]
x3 = np_data[:, 2]
x4 = np_data[:, 3]
y  = np_data[:, 4]

# Extract predictors using regex
terms = re.findall(r'beta_\d+\s+([X\d]+)', Y)
var_map = {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4}
predictors = []

for term in terms:
    var_sequence = re.findall(r'X\d', term)
    var_counts = Counter(var_sequence)
    expr = reduce(lambda acc, var: acc * (var_map[var] ** var_counts[var]), var_counts, np.ones_like(x1))

    predictors.append(expr)

z1, z2, z3, z4 = predictors

# Fit the regression model
X = np.column_stack((np.ones(len(x1)), z1, z2, z3, z4))
model = sm.OLS(y, X).fit()

# Compute XᵀX and inv(XᵀX)
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)

# Covariance matrix of β
# ---------------------------
# The covariance matrix for the β–estimates is given by:
#           Cov(β) = σ² · (XᵀX)⁻¹
# where σ² is the estimated variance of the residuals.
sigma2 = model.mse_resid
cov_beta = sigma2 * XTX_inv


df = int(model.df_resid)  # degrees of freedom for error
t_crit = stats.t.ppf(1-alpha/2, df)

# In model.params, index: 0 = intercept, 1 = X1, 2 = X2, 3 = X3, 4 = X4
beta_x = model.params[ci_beta_x[1]]
se_beta_x = model.bse[ci_beta_x[1]]
if ci_beta_x[0] == "U":
  CI_beta_x = beta_x + abs(t_crit) * se_beta_x
elif ci_beta_x[0] == "L":
  CI_beta_x = beta_x - abs(t_crit) * se_beta_x

p_Val_beta_y = model.pvalues[p_value_beta_y]

# Print results
print("\n******* ANSWERS TO SUBMIT *******\n")
print(f"Element {XTX_index} of XᵀX : {XTX[XTX_idx]:.8f}")
print(f"Element {XTX_inverse_index} of (XᵀX)⁻¹ : {XTX_inv[XTX_inverse_idx]:.8f}")
print(f"Element {Covariance_beta_index} of Cov(β) : {cov_beta[Covariance_beta_idx]:8f}")
print("Lower" if ci_beta_x[0] == "L" else "Upper", f"bound for β{to_subscript(ci_beta_x[1])} at given alpha: {CI_beta_x:.8f}")
print(f"p-value for β{to_subscript(p_value_beta_y)}: {p_Val_beta_y:.6e}")





# Printing Complete Matrices
print("\n******* COMPLETE MATRICES *******\n")
print("XᵀX Matrix:")
print(XTX, "\n")

print("\nInverse of XᵀX:")
print(XTX_inv, "\n")

print("\nCovariance matrix of β:")
print(cov_beta, "\n")


# Extract values
betas = model.params
se = model.bse
p_vals = model.pvalues

# Compute confidence intervals manually
lower_bounds = betas - abs(t_crit) * se
upper_bounds = betas + abs(t_crit) * se

# Display results
print(f"{'Parameter':>10} | {'Beta':>12} | {'SE':>10} | {'p-value':>10} | {'Lower Bound':>15} | {'Upper Bound':>15}")
print("-" * 100)
for i in range(len(betas)):
    print(f"β{i:<9} | {betas[i]:>12.6f} | {se[i]:>10.6f} | {p_vals[i]:>10.6e} | {lower_bounds[i]:>15.6f} | {upper_bounds[i]:>15.6f}")
