import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('classic')

def set_plot_style():
    plt.style.use('classic')
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['font.size'] = 16
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['xtick.minor.size'] = 1.5
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.minor.size'] = 1.5
    plt.rcParams['figure.facecolor'] = "f5f5f5"
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['image.cmap'] = 'plasma'
colors = ['C3', 'C4', 'C5', 'C0', 'C1', 'C6', 'gold', 'r']

def get_tables_per_dimension(df):
    """
    Takes the main results DataFrame and generates a separate summary table 
    for each dimension, with methods as rows.
    """
    columns_to_keep = [
        'Method', 
        'd', 
        'Avg Relative Loss (Pooled across Res)', 
        'Avg Relative Loss StdError (Pooled)',
        'Win Fraction (Pooled across Res)', 
        'Win Fraction StdError (Pooled)'
    ]
    
    df_pooled = df[columns_to_keep].drop_duplicates(subset=['Method', 'd']).copy()
    
    df_pooled = df_pooled.rename(columns={
        'Avg Relative Loss (Pooled across Res)': 'avg_loss_d',
        'Avg Relative Loss StdError (Pooled)': 'avg_loss_d_std',
        'Win Fraction (Pooled across Res)': 'Win Fraction',
        'Win Fraction StdError (Pooled)': 'Win Fraction std'
    })
    
    tables_by_d = {}
    unique_dims = sorted(df_pooled['d'].unique())
    
    for d_val in unique_dims:
        df_d = df_pooled[df_pooled['d'] == d_val].drop(columns=['d'])
        
        df_d = df_d.sort_values(by='avg_loss_d').reset_index(drop=True)
        
        tables_by_d[d_val] = df_d
        
        means = df_d['avg_loss_d'].values
        index_of_baseline_rand = np.where(df_d['Method'] == 'ran')[0][0]
        baseline_mean = means[index_of_baseline_rand]
        stds = df_d['avg_loss_d_std'].values
        z_values = (baseline_mean - means) / np.sqrt(stds**2 + stds[index_of_baseline_rand]**2)
        df_d['z_value'] = z_values
    return tables_by_d

set_plot_style()
np.random.seed(42)

data = pd.read_csv('analysis_results.csv')
print(data.head())
Methods = data['Method'].unique()
print(Methods)
methods_names = ['GridRandom', 'Halton', 'LHS', 'Random', 'Grid', 'RSA', 'Sobol', 'Hyperuniform']
Dimensions = data['d'].unique()
Resolutions = data['res'].unique()

fig, ax = plt.subplots(1, len(Dimensions), figsize=(6*len(Dimensions), 5))

for method_index, method in enumerate(Methods):
    method_data = data[data['Method'] == method]
    for index, d in enumerate(Dimensions):
        dim_data = method_data[method_data['d'] == d]
        dim_data = dim_data.sort_values(by='res')

        if method == 'reg':
            ax[index].errorbar(dim_data['res'], dim_data['Ratio'], label=methods_names[method_index], marker='o', capsize=5, lw=2, color=colors[method_index])
        else:
            ax[index].errorbar(dim_data['res'], dim_data['Ratio'], yerr=dim_data['Error Fraction Ratio'], label=methods_names[method_index], marker='o', capsize=5, lw=2, color=colors[method_index], ls='-')
            ax[index].fill_between(dim_data['res'], dim_data['Ratio'] - dim_data['Error Fraction Ratio'], dim_data['Ratio'] + dim_data['Error Fraction Ratio'], alpha=0.2,color=colors[method_index])
        
        ax[index].set(xlim=(1.9, 5.25), ylim=(0.3, 1.1))
        ax[index].set_facecolor(plt.rcParams['figure.facecolor'])
        ax[index].set(xscale='log')
        ax[index].set_xticks([2, 3, 4, 5])
        ax[index].set_xticklabels([r'$2^{{{}}}$'.format(d), r'$3^{{{}}}$'.format(d), r'$4^{{{}}}$'.format(d), r'$5^{{{}}}$'.format(d)])
for index, d in enumerate(Dimensions):
    ax[index].text(0.05, 0.05, r'${}$'.format(d) + r'$D$', transform=ax[index].transAxes, fontsize=20, verticalalignment='bottom', horizontalalignment='left', fontstyle='normal')
ax[-1].legend(loc=(0.025,0.475),ncol=2, fontsize=13)
ax[1].set_xlabel('Number of sampled points')
ax[0].set_ylabel('scaled loss relative to\ngrid search')
ax[0].set(ylim=(0.6, 1.1))
ax[1].set(ylim=(0.6, 1.1))

fig.suptitle(r'Comparison for LightGBM in $2-4$ dimensions', fontsize=20, y=1.03)
fig.savefig('ratio_plot.png', dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(1, len(Dimensions), figsize=(6*len(Dimensions), 5))
Methods_plot = ['reg', 'ran', 'true_hyper']
methods_names = ['Grid', 'Random', 'Hyperuniform']
colors = ['C1', 'C0', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
for method_index, method in enumerate(Methods_plot):
    method_data = data[data['Method'] == method]
    for index, d in enumerate(Dimensions):
        dim_data = method_data[method_data['d'] == d]
        dim_data = dim_data.sort_values(by='res')
        if method == 'reg':
            ax[index].errorbar(dim_data['res'], dim_data['Ratio'], label=methods_names[method_index], marker='o', capsize=5, lw=2, color=colors[method_index])
        else:
            ax[index].errorbar(dim_data['res'], dim_data['Ratio'], yerr=dim_data['Error Fraction Ratio'], label=methods_names[method_index], marker='o', capsize=5, lw=2, color=colors[method_index], ls='-')
            ax[index].fill_between(dim_data['res'], dim_data['Ratio'] - dim_data['Error Fraction Ratio'], dim_data['Ratio'] + dim_data['Error Fraction Ratio'], alpha=0.2,color=colors[method_index])
        
        ax[index].set(xlim=(1.9, 5.25), ylim=(0.3, 1.1))
        ax[index].set_facecolor(plt.rcParams['figure.facecolor'])
        ax[index].set(xscale='log')
        ax[index].set_xticks([2, 3, 4, 5])
        ax[index].set_xticklabels([r'$2^{{{}}}$'.format(d), r'$3^{{{}}}$'.format(d), r'$4^{{{}}}$'.format(d), r'$5^{{{}}}$'.format(d)])
for index, d in enumerate(Dimensions):
    ax[index].text(0.05, 0.05, r'${}$'.format(d) + r'$D$', transform=ax[index].transAxes, fontsize=20, verticalalignment='bottom', horizontalalignment='left', fontstyle='normal')
ax[0].set(ylim=(0.6, 1.1))
ax[1].set(ylim=(0.6, 1.1))
ax[0].legend(loc='lower right',ncol=1, fontsize=13)
ax[1].set_xlabel('Number of sampled points')
ax[0].set_ylabel('scaled loss relative to\ngrid search')

fig.suptitle(r'Comparison for LightGBM in $2-4$ dimensions', fontsize=20, y=1.03)
fig.savefig('ratio_plot_hyper_uniform.png', dpi=300, bbox_inches='tight')

dimension_tables = get_tables_per_dimension(data)

for d_val, table in dimension_tables.items():
    print(f"\n{'='*40}")
    print(f" RESULTS FOR DIMENSION: {d_val} ")
    print(f"{'='*40}")
    
    print(table.round(4).to_string(index=False))