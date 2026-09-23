import pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, chi2

with open('total_dict_full.pkl', 'rb') as f:
    total_dict = pickle.load(f)

methods = list(total_dict.keys())
min_of_min_over_methods = total_dict[methods[0]][:, 2].copy()
for method in methods:
    data = total_dict[method]
    mask = data[:, 2] < min_of_min_over_methods
    min_of_min_over_methods[mask] = data[:, 2][mask]

data_dict = {}
for method in methods: 
    data_dict[method] = (total_dict[method][:, 2] / min_of_min_over_methods) - 1


unique_methods = sorted(list(set([key[0] for key in total_dict.keys()])))
largest_d = max([key[1] for key in total_dict.keys()])
smallest_d = min([key[1] for key in total_dict.keys()])
largest_res = max([key[2] for key in total_dict.keys()])
smallest_res = min([key[2] for key in total_dict.keys()])

len_d = (largest_d - smallest_d) + 1
len_res = (largest_res - smallest_res) + 1

means = np.zeros((len(unique_methods), len_d, len_res))
stds = np.zeros((len(unique_methods), len_d, len_res))

medians = np.zeros((len(unique_methods), len_d, len_res))
percentiles_1_sigma = np.zeros((len(unique_methods), len_d, len_res))

frac_wins_d = np.zeros((len(unique_methods), len_d))
frac_wins_unc_d = np.zeros((len(unique_methods), len_d))
avg_loss_d = np.zeros((len(unique_methods), len_d))
avg_loss_d_std = np.zeros((len(unique_methods), len_d))


statistics = {}

for method, d, res in total_dict.keys():
    index_method = unique_methods.index(method)
    idx_d = d - smallest_d
    idx_res = res - smallest_res
    
    arr = data_dict[(method, d, res)]
    means[index_method, idx_d, idx_res] = np.mean(arr)
    stds[index_method, idx_d, idx_res] = np.std(arr) / np.sqrt(len(arr))

    stat, p_value = wilcoxon(arr, y = data_dict[('true_hyper', d, res)], alternative='greater', correction=True, zero_method='wilcox')
    statistics[(method, d, res)] = {
        'statistic': stat,
        'p_value': p_value
    }

combined_stats = {}
keys_stats = list(statistics.keys())
#Here we use fisher's method to combine p-values for the same method and dimension
for method in unique_methods:
    for d in range(smallest_d, largest_d + 1):
        p_values = [statistics[(method, d, res)]['p_value'] for res in range(smallest_res, largest_res + 1) if (method, d, res) in statistics]
        if p_values:
            chi2_stat = -2 * np.sum(np.log(p_values))
            df = 2 * len(p_values)
            combined_p_value = 1 - chi2.cdf(chi2_stat, df)
            combined_stats[(method, d)] = {
                'combined_chi2_stat': chi2_stat,
                'combined_df': df,
                'combined_p_value': combined_p_value
            }
combined_stats_df = pd.DataFrame.from_dict(combined_stats, orient='index')
print(combined_stats_df)

error_frac = stds / means
baseline_idx = 4 #the normal grid search method
ratio = means / means[baseline_idx]
error_frac_ratio = ratio * np.sqrt((error_frac)**2 + (error_frac[baseline_idx])**2)


# Pooling metrics across resolutions (Grouped by d)
for d in range(smallest_d, largest_d + 1):
    idx_d = d - smallest_d
    
    wins_for_d = {method: 0 for method in unique_methods}
    total_trials_for_d = 0
    pooled_losses_for_d = {method: [] for method in unique_methods}
    
    for res in range(smallest_res, largest_res + 1):
        valid_methods = [m for m in unique_methods if (m, d, res) in total_dict]
        if not valid_methods:
            continue
            
        for m in valid_methods:
            pooled_losses_for_d[m].extend(data_dict[(m, d, res)])
            
        stacked_losses = np.vstack([total_dict[(m, d, res)][:, 2] for m in valid_methods])
        winner_indices = np.argmin(stacked_losses, axis=0)
        total_trials_for_d += len(winner_indices)
        
        for i, m in enumerate(valid_methods):
            wins_for_d[m] += np.sum(winner_indices == i)

    for method in unique_methods:
        m_idx = unique_methods.index(method)
        
        if total_trials_for_d > 0:
            p_win = wins_for_d[method] / total_trials_for_d
            frac_wins_d[m_idx, idx_d] = p_win
            frac_wins_unc_d[m_idx, idx_d] = np.sqrt((p_win * (1 - p_win)) / total_trials_for_d)
            
        if pooled_losses_for_d[method]:
            avg_loss_d[m_idx, idx_d] = np.mean(pooled_losses_for_d[method])
            avg_loss_d_std[m_idx, idx_d] = np.std(pooled_losses_for_d[method]) / np.sqrt(len(pooled_losses_for_d[method]))

pooled_error_frac = np.zeros_like(avg_loss_d)
np.divide(avg_loss_d_std, avg_loss_d, out=pooled_error_frac, where=(avg_loss_d != 0))

pooled_ratio = avg_loss_d / avg_loss_d[baseline_idx]

# Here we propagate the standard error using standard error propagation
pooled_error_frac_baseline = pooled_error_frac[baseline_idx]
pooled_error_frac_ratio = pooled_ratio * np.sqrt((pooled_error_frac)**2 + (pooled_error_frac_baseline)**2)

avg_loss_d = pooled_ratio
avg_loss_d_std = pooled_error_frac_ratio

index = pd.MultiIndex.from_product(
    [unique_methods, np.arange(smallest_d, largest_d + 1), np.arange(smallest_res, largest_res + 1)],
    names=['Method', 'd', 'res']
)

frac_wins_expanded = np.repeat(frac_wins_d[:, :, np.newaxis], len_res, axis=2)
frac_wins_unc_expanded = np.repeat(frac_wins_unc_d[:, :, np.newaxis], len_res, axis=2)
avg_loss_d_expanded = np.repeat(avg_loss_d[:, :, np.newaxis], len_res, axis=2)
avg_loss_d_std_expanded = np.repeat(avg_loss_d_std[:, :, np.newaxis], len_res, axis=2)

pd_data = pd.DataFrame({
    'Mean': means.flatten(),
    'Std': stds.flatten(),
    'Error Fraction': error_frac.flatten(),
    'Ratio': ratio.flatten(),
    'Error Fraction Ratio': error_frac_ratio.flatten(),
    'Win Fraction (Pooled across Res)': frac_wins_expanded.flatten(),
    'Win Fraction StdError (Pooled)': frac_wins_unc_expanded.flatten(),
    'Avg Relative Loss (Pooled across Res)': avg_loss_d_expanded.flatten(),
    'Avg Relative Loss StdError (Pooled)': avg_loss_d_std_expanded.flatten()
}, index=index).reset_index()

pd_data.to_csv('analysis_results.csv', index=False)