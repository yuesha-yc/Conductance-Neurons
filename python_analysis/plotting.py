
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

def collect_data(dir):
    import os, json
    trials = []
    for root, _, files in os.walk(dir):
        if "params.json" in files and "results.npz" in files:
            with open(os.path.join(root, "params.json")) as f:
                params = json.load(f)
            data = np.load(os.path.join(root, "results.npz"), allow_pickle=True)
            results = {}
            for k, v in data.items():
                v = np.asarray(v)
                # Treat singleton arrays as scalars
                if v.size == 1:
                    v = float(v)
                results[k] = v
            trials.append((params, results))
    return trials


def plot_xy(trials, param_filter, x, y, l=None):
    """
    Plot y (dependent) vs x (independent), optionally grouped by l (legend).
    Average over other trials with same (x,l).
    """
    # Flatten into DataFrame
    rows = []
    for params, results in trials:
        row = {}
        row.update(params)
        for k, v in results.items():
            if np.ndim(v) == 0 or (isinstance(v, np.ndarray) and v.size == 1):
                row[k] = float(np.ravel(v)[0])
        rows.append(row)
    df = pd.DataFrame(rows)

    # Filter
    for k, v in (param_filter or {}).items():
        df = df[df[k] == v]

    if df.empty:
        print("No matching trials.")
        return

    # Group and average
    group_cols = [x] + ([l] if l else [])
    grouped = df.groupby(group_cols)[y].agg(['mean', 'std']).reset_index()

    fig, ax = plt.subplots()
    # set fig size
    # fig.set_size_inches(3, 2)
    if l:
        cmap = plt.get_cmap('viridis')
        labels = list(grouped[l].unique())

        numeric_values = []
        all_numeric = True
        for key in labels:
            try:
                numeric_values.append(float(key))
            except (TypeError, ValueError):
                all_numeric = False
                break

        color_lookup = {}
        if all_numeric and numeric_values:
            numeric_array = np.asarray(numeric_values, dtype=float)
            if numeric_array.size > 1:
                norm = mcolors.Normalize(vmin=numeric_array.min(), vmax=numeric_array.max())
            else:
                center = numeric_array[0]
                norm = mcolors.Normalize(vmin=center - 0.5, vmax=center + 0.5)
            for key in labels:
                color_lookup[key] = cmap(norm(float(key)))
        else:
            if len(labels) == 1:
                color_lookup[labels[0]] = cmap(0.5)
            else:
                for idx, key in enumerate(labels):
                    color_lookup[key] = cmap(idx / (len(labels) - 1))

        for key, sub in grouped.groupby(l):
            color = color_lookup.get(key, None)
            ax.errorbar(sub[x], sub['mean'], yerr=sub['std'], label=f"{l}={key}", fmt='o--', capsize=3, color=color)
        legend = ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0.0)
        fig.tight_layout()
    else:
        ax.errorbar(grouped[x], grouped['mean'], yerr=grouped['std'], fmt='o--', capsize=3)
        fig.tight_layout()

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.show()


def plot_tv(trials, param_filter, t_key, v_key, l=None, x_show_ratio=0.01):
    """
    Plot v(t) curves, averaged over matching trials.
    Each trial has arrays results[t_key] and results[v_key].
    """
    # Collect relevant trials
    data = []
    for params, results in trials:
        if param_filter and not all(params.get(k) == v for k, v in param_filter.items()):
            continue
        if t_key not in results or v_key not in results:
            continue
        t, v = np.asarray(results[t_key]), np.asarray(results[v_key])
        if len(t) != len(v): 
            continue
        label = params.get(l, "All") if l else "All"
        data.append((label, t, v))
    if not data:
        print("No matching trials.")
        return

    fig, ax = plt.subplots()
    for label in sorted(set(lbl for lbl, _, _ in data)):
        subset = [(t, v) for lbl, t, v in data if lbl == label]
        # Align by interpolation if needed
        base_t = subset[0][0]
        all_v = []
        for t, v in subset:
            if np.allclose(t, base_t):
                all_v.append(v)
            else:
                all_v.append(np.interp(base_t, t, v))
        all_v = np.vstack(all_v)
        mean_v = all_v.mean(axis=0)
        std_v = all_v.std(axis=0)
        ax.plot(base_t, mean_v, label=str(label))
        # only show the last 1% of x-axis
        tlen = len(base_t)
        ax.set_xlim(left=base_t[int(tlen * (1 - x_show_ratio))], right=base_t[-1])
        ax.fill_between(base_t, mean_v - std_v, mean_v + std_v, alpha=0.2)
    if l:
        ax.legend()
        fig.tight_layout(rect=[0, 0, 0.82, 1])
    else:
        fig.tight_layout()
    ax.set_xlabel(t_key)
    ax.set_ylabel(v_key)
    plt.show()
