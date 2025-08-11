import numpy as np
from scipy import stats

alpha_F = 0.0353937 # alpha for F test
alphaStar_Bonferroni = 0.738092 # alpha star for Bonferroni Method
delta = 0.94377 # delta for SAM method

data = [
[8.48778,10.0508],
[8.90633,8.18504],
[7.969,8.71892],
[8.64976,10.3363],
[8.36,9.33817],
[9.74315,9.5753],
[9.55446,9.77323],
[8.16989,9.28318],
[8.92781,11.435],
[9.5348,10.5411],
[8.94829,9.72486],
[7.61398,9.69247],
[9.15229,11.3942],
[9.02974,9.52979],
[8.82869,8.3823],
[7.85773,10.3236],
[7.80447,8.58249],
[7.59989,8.09007],
[7.24856,10.0758],
[8.44769,10.6005],
[9.33795,9.07029],
[7.89766,10.4388],
[7.94487,8.85403],
[7.95508,10.7585],
[7.67142,10.5677],
[8.26368,8.87623],
[9.13435,8.56527],
[9.54613,9.22394],
[8.82604,11.394],
[8.66345,9.59536],
[7.48381,10.2876],
[7.71933,8.6995],
[8.16002,10.0709],
[9.15439,9.23369],
[9.18648,9.34907],
[8.58831,9.32144],
[9.84991,9.6555],
[9.25132,10.4376],
[7.57954,10.7428],
[8.119,10.7697],
[8.78341,10.1543],
[8.78384,8.79289],
[8.40055,7.61897],
[8.93499,8.83689],
[9.22348,10.293],
[8.16939,9.93894],
[10.7441,10.565],
[7.55393,11.927],
[9.35743,10.7319],
[8.20597,9.17611],
[8.46012,9.82595],
[8.29458,8.46481],
[9.2065,10.4218],
[7.84701,8.06527],
[8.41093,9.78263],
[8.26481,9.44524],
[9.22454,9.08903],
[10.314,9.65957],
[9.0049,10.1948],
[8.56812,10.187],
[8.22411,10.4617],
[8.08508,10.5395],
[8.51656,12.0342],
[9.66211,10.856],
[8.36525,10.3345],
[9.5071,10.7977],
[9.18289,9.42761],
[7.61605,9.83321],
[9.84862,10.113],
[9.0618,10.3065],
[8.87269,6.77482],
[9.41498,8.19941],
[8.29273,7.81157],
[8.86378,8.94082],
[7.28958,8.36793]
]



## SWAP indices for SHUFFLING 1
swap_indices_1 = [
[5, 3],
[36, 34],
[51, 25],
[66, 24],
[19, 71],
[53, 49],
[26, 10],
[67, 43],
[33, 53],
[24, 68],
[1, 6],
[26, 36],
[60, 23],
[33, 56],
[46, 12],
[46, 49],
[52, 41],
[68, 26],
[50, 18],
[7, 22],
[22, 28],
[65, 33],
[36, 9],
[32, 7],
[2, 71],
[61, 6],
[55, 11],
[13, 26],
[14, 26],
[67, 63],
[20, 57],
[49, 12],
[65, 1],
[7, 20],
[30, 29],
[41, 14],
[56, 22],
[35, 63],
[17, 7],
[26, 2],
[4, 19],
[69, 8],
[57, 2],
[4, 51],
[36, 29],
[75, 16],
[73, 13],
[53, 18],
[9, 25],
[30, 66],
[31, 44],
[70, 75],
[31, 51],
[10, 7],
[6, 54],
[9, 33],
[31, 37],
[6, 67],
[72, 65],
[46, 34],
[33, 23],
[50, 64],
[26, 18],
[33, 56],
[44, 56],
[11, 16],
[3, 61],
[48, 47],
[47, 37],
[36, 31],
[27, 61],
[42, 68],
[21, 61],
[19, 56],
[36, 9]
]


## SWAP indices for SHUFFLING 2
swap_indices_2 = [
[52, 16],
[39, 57],
[63, 73],
[1, 28],
[57, 25],
[28, 48],
[33, 33],
[1, 19],
[8, 57],
[39, 48],
[26, 2],
[68, 33],
[17, 17],
[73, 50],
[53, 51],
[74, 48],
[41, 9],
[9, 40],
[52, 67],
[2, 40],
[54, 72],
[17, 12],
[10, 69],
[71, 29],
[3, 40],
[43, 74],
[19, 33],
[22, 52],
[54, 24],
[68, 10],
[61, 32],
[30, 10],
[24, 51],
[38, 2],
[49, 48],
[34, 58],
[17, 16],
[61, 30],
[36, 30],
[53, 13],
[54, 11],
[27, 56],
[11, 66],
[31, 62],
[62, 75],
[3, 14],
[75, 42],
[67, 28],
[18, 37],
[11, 35],
[22, 65],
[66, 30],
[20, 29],
[21, 7],
[3, 34],
[23, 36],
[1, 3],
[64, 19],
[36, 39],
[53, 45],
[37, 34],
[4, 8],
[65, 21],
[56, 67],
[48, 63],
[72, 60],
[55, 50],
[8, 9],
[60, 28],
[61, 17],
[51, 62],
[69, 50],
[19, 54],
[30, 47],
[9, 26]
]

import numpy as np
from scipy import stats

# Convert input data to NumPy array
raw_data = np.array(data)

# Convert 1-indexed swap indices to 0-indexed
swap_idx_1 = np.array(swap_indices_1) - 1
swap_idx_2 = np.array(swap_indices_2) - 1

# Parameters
alpha_star = alphaStar_Bonferroni
alpha = alpha_F
delta_cutoff = delta
num_averages = 1
subset_size = 5

# Data dimensions
num_samples = raw_data.shape[0]
num_genes = num_samples // subset_size

# Split data into two variables
x_data = raw_data[:, 0]
y_data = raw_data[:, 1]

# Function to perform value swapping at specific indices
def swap_values(x, y, swap_pairs):
    swapped_x = x.copy()
    swapped_y = y.copy()

    for ix, iy in swap_pairs:
        if ix < len(swapped_x) and iy < len(swapped_y):
            swapped_x[ix], swapped_y[iy] = swapped_y[iy], swapped_x[ix]
    return {'x': swapped_x, 'y': swapped_y}

# Function to perform a two-sample t-test
def perform_ttest(x, y, alpha):
    t_stat, p_val = stats.ttest_ind(x, y, equal_var=True)
    reject_null = 1.0 if p_val < alpha else 0.0
    return {'t_stat': t_stat, 'p_value': p_val, 'reject_null': reject_null}

# Apply swapping based on the provided indices
shuffled_1 = swap_values(x_data, y_data, swap_idx_1)
shuffled_2 = swap_values(x_data, y_data, swap_idx_2)

# Initialize result containers
t_values_orig = np.zeros(num_genes)
t_values_shuf1 = np.zeros(num_genes)
t_values_shuf2 = np.zeros(num_genes)
p_values_orig = np.zeros(num_genes)
p_values_shuf1 = np.zeros(num_genes)

fdr_total = 0
count = 0

# Repeat analysis over multiple averaging passes
for _ in range(num_averages):
    significant_orig = 0
    significant_shuf = 0

    for i in range(num_genes):
        idx_start = i * subset_size
        idx_end = (i + 1) * subset_size

        # Extract subsets for original and shuffled data
        x_orig = x_data[idx_start:idx_end]
        y_orig = y_data[idx_start:idx_end]
        x_shuf1 = shuffled_1['x'][idx_start:idx_end]
        y_shuf1 = shuffled_1['y'][idx_start:idx_end]
        x_shuf2 = shuffled_2['x'][idx_start:idx_end]
        y_shuf2 = shuffled_2['y'][idx_start:idx_end]

        # Original data t-test
        result_orig = perform_ttest(x_orig, y_orig, 2 * alpha)
        t_values_orig[i] = result_orig['t_stat']
        p_values_orig[i] = result_orig['p_value']
        significant_orig += result_orig['reject_null']

        # First shuffled t-test
        result_shuf1 = perform_ttest(x_shuf1, y_shuf1, 2 * alpha)
        t_values_shuf1[i] = result_shuf1['t_stat']
        p_values_shuf1[i] = result_shuf1['p_value']
        significant_shuf += result_shuf1['reject_null']

        # Second shuffled t-test (used for SAM)
        result_shuf2 = perform_ttest(x_shuf2, y_shuf2, 2 * alpha)
        t_values_shuf2[i] = result_shuf2['t_stat']

    # Estimate FDR
    if significant_orig > 0:
        fdr_total += significant_shuf / significant_orig
        count += 1

# Sort t-values and p-values
sorted_t_orig = np.sort(t_values_orig)
sorted_t_shuf1 = np.sort(t_values_shuf1)
sorted_t_shuf2 = np.sort(t_values_shuf2)
sorted_p_orig = np.sort(p_values_orig)
sorted_p_shuf1 = np.sort(p_values_shuf1)

# SAM-based FDR calculation
diff_sam = np.abs(sorted_t_orig - sorted_t_shuf1) / np.sqrt(2)
diff_sam_null = np.abs(sorted_t_shuf1 - sorted_t_shuf2) / np.sqrt(2)
num_significant_sam = np.sum(diff_sam > delta_cutoff)
num_false_sam = np.sum(diff_sam_null > delta_cutoff)
fdr_sam = 100 * (num_false_sam / num_significant_sam)

# Bonferroni-based FDR calculation
threshold = alpha_star / num_genes
num_significant_bonf = np.sum(sorted_p_orig < threshold)
num_false_bonf = np.sum(sorted_p_shuf1 < threshold)
fdr_bonf = 100 * (num_false_bonf / num_significant_bonf)

# Results
print("\nBonferroni method:\n")
print(f"no of H1 in Bonferroni: {num_significant_bonf},\nFDR (%) in Bonferroni: {fdr_bonf:.2f}%\n")

print("SAM method:\n")
print(f"no of H1 in SAM: {num_significant_sam},\nFDR (%) in SAM: {fdr_sam:.2f}%")

print("\nFDR for both Bonferroni and SAM should not be zero,\nif it is coming zero, then most probably it'll be wrong,\npls help us if you find the solution to this problem; (try submitting between 20% to 50%)")
