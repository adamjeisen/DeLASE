from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np

def plot_AIC(results_dict, matrix_size=None, r=None, figsize=(24, 8), title='', picked_color='orange', labelsize=20, tick_param_size=15, title_size=22, marker_size=24):
    matrix_size_vals = results_dict[list(results_dict.keys())[0]].index.get_level_values('matrix_size').unique().values
    r_vals = results_dict[list(results_dict.keys())[0]].index.get_level_values('r').unique().values
    if matrix_size is not None:
        m_index = np.where(matrix_size_vals == matrix_size)[0][0]
    else:
        m_index = None
    if r is not None:
        r_index = np.where(r_vals == r)[0][0]
    else:
        r_index = None
    
    AIC = np.zeros((len(matrix_size_vals), len(r_vals)))
    counts = np.zeros(AIC.shape)
    for key, results in results_dict.items():
        for i, matrix_size in enumerate(matrix_size_vals):
            for j, r in enumerate(r_vals):
                row = results[np.logical_and(results.index.get_level_values('matrix_size') == matrix_size, results.index.get_level_values('r') == r)]
                if len(row) >= 1:
                    row = row.iloc[0]
                    if 'AICs' in row:
                        AIC[i, j] += np.sum(row.AICs)
                        counts[i, j] += len(row.AICs)
                    else: # 'AIC' in row
                        AIC[i, j] += row.AIC
                        counts[i, j] += 1
    AIC[counts == 0] = np.Inf
    counts[counts == 0] = 1
    AIC /= counts
    m_index_, r_index_ = np.unravel_index(AIC.argmin(), shape=AIC.shape)
    if m_index is None:
        m_index = m_index_
    if r_index is None:
        r_index = r_index_
    
    AIC[AIC == np.Inf] = np.nan

    plt.figure(figsize=figsize)

    norm = Normalize(vmin=AIC[~np.isnan(AIC)].min(),vmax=AIC[~np.isnan(AIC)].max())

    plt.imshow(AIC, norm=norm, aspect='auto')
    plt.yticks(np.arange(len(matrix_size_vals)), [f"{matrix_size}" for matrix_size in matrix_size_vals])
    plt.ylabel("matrix size", fontsize=labelsize)
    plt.xticks(np.arange(len(r_vals)), [f"{r}" for r in r_vals])
    plt.xlabel('rank', fontsize=labelsize)
    ax = plt.gca()
    ax.tick_params(labelsize=tick_param_size)
    # ax.set_title(f'AIC\nmatrix size = {matrix_size_vals[m_index]}, r = {r_vals[r_index]}' if len(title) == 0 else f"{title}\nmatrix size = {matrix_size_vals[m_index]}, r = {r_vals[r_index]}", fontsize=title_size)
    ax.scatter(r_index, m_index, c=picked_color, s=marker_size)
    plt.colorbar(ax=ax, label='AIC')

    plt.tight_layout()
    plt.show()