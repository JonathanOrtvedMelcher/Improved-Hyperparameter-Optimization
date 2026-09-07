import csv
import io
import os
import pickle
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

try:
    from scipy.stats import wilcoxon  # pyright: ignore[reportMissingImports]
except ImportError:
    wilcoxon = None

plt.style.use("classic")

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"

DEFAULT_COLORS = {
    "grid": "green",
    "random": "blue",
    "gridrandom": "black",
    "RSA": "orange",
    "latin": "teal",
    "hyperuniform": "red",
    "sobol": "navy",
    "halton": "olive",
}


DEFAULT_MARKERS = {
    "grid": "o",
    "random": "^",
    "gridrandom": "D",
    "RSA": "h",
    "latin": "v",
    "sobol": "P",
    "halton": "p",
    "hyperuniform": "s",
}

MARKER_CYCLE = ["o", "^", "s", "D", "v", "P", "*", "X", "p", "h", "d", "<", ">"]

HYPERUNIFORM_STEALTH_MARKERS = {
    0.49: "*",
    0.4: "s",
    0.38: "X",
    0.3: "X",
    0.25: "P",
    0.13: "d",
}

DEFAULT_LABELS = {
    "grid": "Grid",
    "random": "Random",
    "gridrandom": "Gridrandom",
    "RSA": "RSA",
    "latin": "LHS",
    "hyperuniform": "Hyperuniform",

    "hyperuniform_0.49": "Hyperuniform_0.49",
}

for stealth in (0.4, 0.3, 0.49, 0.38, 0.25, 0.13):
    DEFAULT_COLORS[f"hyperuniform_{stealth}"] = "red"
    DEFAULT_MARKERS[f"hyperuniform_{stealth}"] = HYPERUNIFORM_STEALTH_MARKERS[stealth]
    DEFAULT_LABELS[f"hyperuniform_{stealth}"] = f"Hyperuniform_{stealth}"


DEFAULT_YLIM = {
    2: (0.2, 1.65),
    3: (0.2, 1.1),
    4: (0.2, 1.05),
    5: (0.2, 1.05),
}

PALETTE = [
    "black", "red", "magenta", "olive", "yellow", "olivedrab",
    "darkolivegreen", "forestgreen", "turquoise", "chocolate", "teal",
    "deepskyblue", "slategray", "navy", "blueviolet", "violet",
    "purple", "deeppink", "crimson",
]

# Wilcoxon reference: highest-stealth hyperuniform per dimension (override via hyperuniform_reference).
DEFAULT_HYPERUNIFORM_REFERENCE = {
    2: "hyperuniform_0.4",
}
DEFAULT_HYPERUNIFORM_REFERENCE_OTHER = "hyperuniform_0.4"

FUNC_DISPLAY_NAMES = {
    "branin": "Branin",
    "generic": "Generic",
    "langermann": "Langermann",
    "rastrigin": "Rastrigin",
    "rastrigin_reduced": "Rastrigin (reduced)",
    "schwefel": "Schwefel",
    "styblinski_tang": "Styblinski-Tang",
    "zakharov": "Zakharov",
    "simple": "Simple",
}

# Legend sits on axs[row, 1] (right column: 3D if row=0, 5D if row=1).
# loc is "upper right" or "lower right". Edit per function as needed.
_DEFAULT_LEGEND_LOCATION = {"row": 0, "loc": "upper right"}
FUNC_LEGEND_LOCATION = {
    "branin": {"row": 1, "loc": "lower right"},
    "branin_rotated": {"row": 1, "loc": "lower right"},
    "generic": {"row": 0, "loc": "upper right"},
    "langermann": {"row": 0, "loc": "upper right"},
    "rastrigin": {"row": 0, "loc": "upper right"},
    "rastrigin_reduced": {"row": 0, "loc": "upper right"},
    "schwefel": {"row": 0, "loc": "upper right"},
    "schwefel_rotated": {"row": 1, "loc": "lower right"},
    "styblinski_tang": {"row": 0, "loc": "upper right"},
    "zakharov": {"row": 1, "loc": "lower right"},
    "simple": {"row": 1, "loc": "lower right"},
}


def _func_display_name(func_name):
    """Pretty name for plot titles; falls back to title case if unlisted."""
    if func_name in FUNC_DISPLAY_NAMES:
        return FUNC_DISPLAY_NAMES[func_name]
    return str(func_name).replace("_", " ").title()


def _usual_pattern_label(name, label_map):
    """Legend spelling for a pattern, including `{pattern}_rotated` -> `rotated {label}`."""
    if name in label_map:
        return label_map[name]
    if name.endswith("_rotated"):
        base = name[: -len("_rotated")]
        return "rotated {}".format(_usual_pattern_label(base, label_map))
    if name.startswith("quasi_"):
        return "Quasiperiodic ({})".format(name.replace("quasi_", "").replace("star", " star"))
    return name.replace("_", " ").title()


def _func_legend_location(func_name):
    """Return (row, loc) for the figure legend; column is always 1."""
    spec = FUNC_LEGEND_LOCATION.get(func_name, _DEFAULT_LEGEND_LOCATION)
    return spec.get("row", 0), spec.get("loc", "upper right")


def load_run_results(run_dir, d):
    """Load results_d_{d}.pkl from a run directory, with fallback to legacy pickles.

    Returns None if no results file exists for this dimension.
    """
    run_dir = os.path.abspath(run_dir)
    full_path = os.path.join(run_dir, "results_d_{}.pkl".format(d))
    if os.path.isfile(full_path):
        with open(full_path, "rb") as f:
            return pickle.load(f)

    prefix = os.path.join(run_dir, "")
    scores_path = prefix + "Scores_d_{}.pkl".format(d)
    numbers_path = prefix + "Numbers_d_{}.pkl".format(d)
    if not os.path.isfile(scores_path):
        scores_path = prefix + "Scores_d_{}.txt".format(d)
    if not os.path.isfile(numbers_path):
        numbers_path = prefix + "Numbers_d_{}.txt".format(d)

    if not os.path.isfile(scores_path) or not os.path.isfile(numbers_path):
        return None

    with open(scores_path, "rb") as f:
        everything_scores = pickle.load(f)
    with open(numbers_path, "rb") as f:
        everything_numbers = pickle.load(f)

    pattern_names = sorted({name for block in everything_scores for name in block})
    return {
        "metadata": {"d": d, "pattern_names": pattern_names},
        "everything_scores": everything_scores,
        "everything_numbers": everything_numbers,
        "everything_means": None,
        "everything_stderr": None,
        "by_grid_size": None,
    }


def _resolve_styles(patterns, colors=None, labels=None):
    """Return dicts name -> color, marker, legend label."""
    color_map = dict(DEFAULT_COLORS)
    label_map = dict(DEFAULT_LABELS)
    marker_map = dict(DEFAULT_MARKERS)

    if colors is not None:
        if isinstance(colors, dict):
            color_map.update(colors)
        else:
            for name, color in zip(patterns, colors):
                color_map[name] = color

    if labels is not None:
        if isinstance(labels, dict):
            label_map.update(labels)
        else:
            for name, label in zip(patterns, labels):
                label_map[name] = label

    for i, name in enumerate(patterns):
        if name not in color_map:
            color_map[name] = PALETTE[i % len(PALETTE)]
        if name not in marker_map:
            marker_map[name] = MARKER_CYCLE[i % len(MARKER_CYCLE)]
        if name not in label_map:
            label_map[name] = _usual_pattern_label(name, label_map)

    return color_map, marker_map, label_map


def _patterns_for_dimension(patterns, dim):
    """Return the pattern list for a given dimension."""
    if patterns is None:
        return ["grid", "random", "hyperuniform"]
    if isinstance(patterns, dict):
        if dim not in patterns:
            raise ValueError("No pattern list provided for dimension {}".format(dim))
        return list(patterns[dim])
    return list(patterns)


def _empty_plot_summary():
    return {
        "means": {},
        "stderr": {},
        "numbers": {},
        "rel_means": {},
        "rel_stderr": {},
        "largest_idx": 0,
        "wilcoxon_summary": {},
        "wilcoxon_reference": None,
        "wilcoxon_patterns": [],
        "metadata": {},
    }


def _style_empty_panel(ax, d, position="down", y_lim=None):
    """Apply the usual panel chrome with no series (missing results file)."""
    ax.set_title("{}D".format(d), fontsize=30, y=0.84, x=0.1)
    ax.set_xscale("log")
    x_lo = max(2**d * 0.9, 1) if d <= 5 else 1
    x_hi = max(x_lo * 20, 10)
    ax.set_xlim((x_lo, x_hi))
    if y_lim is None:
        y_lim = DEFAULT_YLIM.get(d)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.tick_params(labelsize=25)
    if position == "up":
        ax.tick_params(
            "x",
            bottom=True,
            top=True,
            labelbottom=False,
            labeltop=True,
            labelsize=25,
        )
    ax.yaxis.set_label_coords(1.05, 0.5)
    ax.set_axisbelow(True)
    ax.set_facecolor("whitesmoke")


def _ordered_pattern_union(dimensions, patterns, summaries_by_dim):
    """Return pattern names in first-seen order across dimensions."""
    ordered = []
    for dim in dimensions:
        if dim not in summaries_by_dim:
            continue
        if isinstance(patterns, dict):
            dim_plot = patterns.get(dim, [])
        elif patterns is None:
            dim_plot = []
        else:
            dim_plot = list(patterns)
        extras = summaries_by_dim[dim].get("wilcoxon_patterns") or []
        for name in list(dim_plot) + list(extras):
            if name not in ordered:
                ordered.append(name)
    return ordered


def _wilcoxon_patterns_for_dimension(wilcoxon_patterns, plot_patterns, dim, reference):
    """Patterns to include in the Wilcoxon test; defaults to the plotted set."""
    if wilcoxon_patterns is None:
        names = list(plot_patterns)
    else:
        names = _patterns_for_dimension(wilcoxon_patterns, dim)
    if reference is not None and reference not in names:
        names.append(reference)
    return names


def _hyperuniform_reference_for_dim(d, hyperuniform_reference=None, patterns=None):
    """
    Return the hyperuniform pattern name used as the Wilcoxon reference for dimension d.

    hyperuniform_reference may be a single pattern name or a dict mapping dimension -> name.
    If not provided, uses hyperuniform_0.49 for d=2 and hyperuniform_0.4 otherwise.
    """
    if hyperuniform_reference is not None:
        if isinstance(hyperuniform_reference, dict):
            if d in hyperuniform_reference:
                return hyperuniform_reference[d]
        else:
            return hyperuniform_reference

    if d in DEFAULT_HYPERUNIFORM_REFERENCE:
        return DEFAULT_HYPERUNIFORM_REFERENCE[d]
    return DEFAULT_HYPERUNIFORM_REFERENCE_OTHER


def _means_stderr_from_results(results, patterns):
    """Extract mean, stderr, num_points per pattern from saved summaries or raw scores."""
    if results.get("everything_means") is not None:
        means = {p: list(results["everything_means"][p]) for p in patterns if p in results["everything_means"]}
        stderr = {p: list(results["everything_stderr"][p]) for p in patterns if p in results["everything_stderr"]}
        numbers = {p: list(results["everything_numbers"][p]) for p in patterns if p in results["everything_numbers"]}
        return means, stderr, numbers

    means = {p: [] for p in patterns}
    stderr = {p: [] for p in patterns}
    numbers = {p: [] for p in patterns}

    for block in results["everything_scores"]:
        for p in patterns:
            if p not in block or not block[p]:
                continue
            scores = block[p]
            means[p].append(float(np.mean(scores)))
            stderr[p].append(float(np.std(scores) / np.sqrt(len(scores))))
            if results["everything_numbers"] and p in results["everything_numbers"]:
                idx = len(means[p]) - 1
                nums = results["everything_numbers"][p]
                numbers[p].append(nums[idx] if idx < len(nums) else len(scores))
            else:
                numbers[p].append(len(scores))

    return means, stderr, numbers


def _apply_loss_offset(means, d, loss_offset_per_dim):
    """Shift all pattern means by loss_offset_per_dim * d (stderr unchanged)."""
    if not loss_offset_per_dim:
        return means
    shift = loss_offset_per_dim * d
    return {name: [value + shift for value in values] for name, values in means.items()}


def ylim_from_data(data, pad=1.1, stderr=None, d=None):
    """Return (ymin, ymax) at `pad` times the min/max plotted values.

    Min is usually negative (strategy better than grid), so ``pad * ymin``
    expands downward; ``pad * ymax`` expands upward. Pass a list, or a
    dict of lists (e.g. rel_means). Optional stderr is included so error
    bars are not clipped.

    Examples
    --------
    ax.set_ylim(*ylim_from_data(rel_means, stderr=rel_stderr))
    ax.set_ylim(*ylim_from_data(rel_means, pad=1.2))
    """
    if isinstance(data, dict):
        ys = []
        for name, values in data.items():
            # if not (name == "grid" or (name == "latin" and d == 2)):
            # if not (name == "grid"):
                errs = (stderr or {}).get(name) or [0.0] * len(values)
                for y, e in zip(values, errs):
                    ys.extend((y - e, y + e))
    else:
        raise ValueError
        ys = list(data)
        if stderr is not None:
            for y, e in zip(data, stderr):
                ys.extend((y - e, y + e))
    if not ys:
        return (0.0, 1.0)
    ymin, ymax = min(ys), max(ys)
    lo = ymin * pad if ymin < 0 else ymin
    hi = ymax * pad if ymax > 0 else ymax
    if lo == hi:
        span = abs(lo) * (pad - 1.0) or 1.0
        lo, hi = lo - span, hi + span
    return (lo, hi)


def _relative_to_baseline(means, stderr, baseline):
    """(strategy − grid) / |grid| per resolution. Grid sits at y=0."""
    rel_means = {}
    rel_stderr = {}
    base = means[baseline]
    base_err = stderr[baseline]
    for name in means:
        rel_means[name] = []
        rel_stderr[name] = []
        for j in range(len(means[name])):  # iterate over resolutions
            s = means[name][j]
            g = base[j]
            m = (s - g) / abs(g)
            rel_means[name].append(m)
            if name == baseline:
                rel_stderr[name].append(0.0)
            else:
                # |s/g| * sqrt((σ_s/s)^2 + (σ_g/g)^2); written without /s so s=0 is safe
                rel_stderr[name].append(
                    float(np.sqrt((stderr[name][j] / g) ** 2 + (s * base_err[j] / g ** 2) ** 2))
                )
    for j in range(len(base)):
        rel_means[baseline][j] = 0.0
    return rel_means, rel_stderr


def _largest_resolution_index(results):
    """Return the index corresponding to the largest stored resolution."""
    by_grid_size = results.get("by_grid_size") or []
    if by_grid_size:
        return max(
            range(len(by_grid_size)),
            key=lambda i: by_grid_size[i].get(
                "reference_num_points",
                by_grid_size[i].get("target_points", float("-inf")),
            ),
        )
    return 0


def _paired_scores_from_results(results, strategy, reference, resolution_index):
    """Collect matched per-trial losses for one strategy/reference pair at one resolution."""
    blocks = results.get("everything_scores") or []
    if resolution_index >= len(blocks):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    block = blocks[resolution_index]
    if strategy not in block or reference not in block:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    n_pairs = min(len(block[strategy]), len(block[reference]))
    if n_pairs <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    return (
        np.asarray(block[strategy][:n_pairs], dtype=float),
        np.asarray(block[reference][:n_pairs], dtype=float),
    )


def _paired_wilcoxon_vs_reference(results, patterns, reference):
    """Run one-sided paired Wilcoxon tests vs. one reference at the largest resolution."""
    if wilcoxon is None or reference not in patterns:
        return {}
    resolution_index = _largest_resolution_index(results)
    reference_summary = {}
    for strategy in patterns:
        if strategy == reference:
            continue
        strategy_scores, reference_scores = _paired_scores_from_results(
            results, strategy, reference, resolution_index
        )
        if strategy_scores.size == 0:
            continue

        diffs = strategy_scores - reference_scores
        nonzero_diffs = diffs[diffs != 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.divide(
                strategy_scores,
                reference_scores,
                out=np.full(strategy_scores.shape, np.nan, dtype=float),
                where=reference_scores != 0,
            )

        summary = {
            "n_pairs": int(strategy_scores.size),
            "median_delta": float(np.median(diffs)),
            "median_ratio": float(np.nanmedian(ratios)),
        }
        if nonzero_diffs.size == 0:
            summary["note"] = "all paired differences are zero"
        else:
            statistic, pvalue = wilcoxon(
                strategy_scores,
                reference_scores,
                alternative="greater",
                zero_method="wilcox",
            )
            summary["statistic"] = float(statistic)
            summary["pvalue"] = float(pvalue)
        reference_summary[strategy] = summary
    return {reference: reference_summary} if reference_summary else {}


def _print_summary(
    d,
    patterns,
    means,
    rel_means,
    largest_idx=0,
    wilcoxon_summary=None,
    baseline="grid",
):
    print("---------------------------------------------------------")
    print("{} dimensions - results from run\n".format(d))

    if baseline in means and "hyperuniform" in rel_means:
        impr_g = 100 * np.mean(
            [(means["hyperuniform"][j] - means[baseline][j]) / means[baseline][j] for j in range(len(means[baseline]))]
        )
        impr_r = 100 * np.mean(
            [(means["hyperuniform"][j] - means["random"][j]) / means["random"][j] for j in range(len(means["random"]))]
            if "random" in means
            else float("nan")
        )
        print("Improvement of hyperuniform over grid search in %:", -impr_g)
        print("Improvement of hyperuniform over random search in %:", -impr_r)

    print("(Loss − grid) / |grid| at the largest resolution:")
    for p in patterns:
        values = rel_means.get(p, [])
        if 0 <= largest_idx < len(values):
            print(" ", p, values[largest_idx])
    if wilcoxon_summary:
        print("\nPaired Wilcoxon signed-rank tests (largest resolution, alternative: reference < strategy):")
        for reference in wilcoxon_summary:
            print(" Against {}:".format(reference))
            for p in wilcoxon_summary[reference]:
                summary = wilcoxon_summary[reference][p]
                line = "  {}: n={}, median delta={}, median ratio={}".format(
                    p,
                    summary["n_pairs"],
                    summary["median_delta"],
                    summary["median_ratio"],
                )
                if "pvalue" in summary:
                    line += ", statistic={}, p-value={}".format(
                        summary["statistic"],
                        summary["pvalue"],
                    )
                if "note" in summary:
                    line += " ({})".format(summary["note"])
                print(line)
    print()


def _format_table_float(value):
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return ""
    return "{:.6g}".format(float(value))


def _extract_largest_resolution_stats(summary, pattern):
    largest_idx = summary.get("largest_idx", 0)
    rel_means = summary.get("rel_means", {}).get(pattern, [])
    rel_stderr = summary.get("rel_stderr", {}).get(pattern, [])
    mean_value = rel_means[largest_idx] if 0 <= largest_idx < len(rel_means) else None
    stderr_value = rel_stderr[largest_idx] if 0 <= largest_idx < len(rel_stderr) else None
    return mean_value, stderr_value


def _extract_wilcoxon_pvalue(summary, reference, pattern):
    ref_summary = summary.get("wilcoxon_summary", {}).get(reference, {})
    pattern_summary = ref_summary.get(pattern, {})
    return pattern_summary.get("pvalue")


def _wilcoxon_reference_for_summary(summary, dim, hyperuniform_reference=None, patterns=None):
    """Resolve the Wilcoxon reference pattern for one dimension."""
    if summary.get("wilcoxon_reference"):
        return summary["wilcoxon_reference"]
    dim_patterns = _patterns_for_dimension(patterns, dim) if patterns is not None else None
    return _hyperuniform_reference_for_dim(dim, hyperuniform_reference, dim_patterns)


def _build_summary_table(patterns, dimensions, summaries_by_dim, hyperuniform_reference=None):
    row_patterns = _ordered_pattern_union(dimensions, patterns, summaries_by_dim)
    headers = ["Pattern"]
    for dim in dimensions:
        if dim not in summaries_by_dim:
            continue
        ref = _wilcoxon_reference_for_summary(
            summaries_by_dim[dim], dim, hyperuniform_reference, patterns
        )
        headers.extend(
            [
                "Mean loss ({}D)".format(dim),
                "p-value vs {} ({}D)".format(ref, dim),
            ]
        )

    rows = []
    for pattern in row_patterns:
        value_row = [pattern]
        error_row = [""]
        for dim in dimensions:
            if dim not in summaries_by_dim:
                continue
            summary = summaries_by_dim[dim]
            ref = _wilcoxon_reference_for_summary(
                summary, dim, hyperuniform_reference, patterns
            )
            mean_value, stderr_value = _extract_largest_resolution_stats(summary, pattern)
            pvalue = ""
            if pattern != ref:
                pvalue = _format_table_float(_extract_wilcoxon_pvalue(summary, ref, pattern))
            value_row.extend(
                [
                    _format_table_float(mean_value),
                    pvalue,
                ]
            )
            error_row.extend(
                [
                    _format_table_float(stderr_value),
                    "",
                ]
            )
        rows.append(value_row)
        rows.append(error_row)
    return headers, rows


def _table_to_csv_text(headers, rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows)
    return buffer.getvalue().strip()


def _print_summary_table(
    patterns, dimensions, summaries_by_dim, csv_path=None, hyperuniform_reference=None
):
    headers, rows = _build_summary_table(
        patterns, dimensions, summaries_by_dim, hyperuniform_reference
    )
    table_text = _table_to_csv_text(headers, rows)
    print("\nSummary table (CSV format):")
    print(table_text)
    print()

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print("Saved summary table to {}".format(csv_path))
        print()


def count_best_strategy_trials(
    run_dirs,
    dimensions=(2, 3, 4, 5),
    patterns=None,
    csv_path=None,
    print_stats=True,
):
    """Count how often each strategy has the lowest raw loss.

    One comparison is one trial at one resolution in one dimension. The
    grand total is therefore (trials) × (resolutions) × (dimensions 2–5),
    using whatever was actually stored (typically 10,000 trials each).

    A strategy is counted as best if its loss equals the minimum among the
    considered patterns on that comparison (ties increment every tied
    strategy). Uses raw trial scores, not the figure's relative-loss metric.
    """
    if isinstance(run_dirs, str):
        run_dirs = {dimensions[0]: run_dirs}

    ordered = []
    wins = {}
    n_comparisons = 0
    n_tied = 0
    by_dim = {}

    for dim in dimensions:
        if dim not in run_dirs:
            continue
        dim_patterns = _patterns_for_dimension(patterns, dim)
        results = load_run_results(run_dirs[dim], dim)
        if results is None:
            continue
        blocks = results.get("everything_scores") or []
        dim_wins = {p: 0 for p in dim_patterns}
        dim_comparisons = 0
        dim_tied = 0

        for block in blocks:
            available = [p for p in dim_patterns if p in block and block[p]]
            if len(available) < 2:
                continue
            n_trials = min(len(block[p]) for p in available)
            if n_trials <= 0:
                continue
            mat = np.column_stack(
                [np.asarray(block[p][:n_trials], dtype=float) for p in available]
            )
            mins = np.min(mat, axis=1)
            is_best = mat == mins[:, None]
            n_best = is_best.sum(axis=1)
            dim_tied += int(np.sum(n_best > 1))
            dim_comparisons += int(n_trials)
            for i, name in enumerate(available):
                dim_wins[name] = dim_wins.get(name, 0) + int(is_best[:, i].sum())

        by_dim[dim] = {
            "wins": dim_wins,
            "n_comparisons": dim_comparisons,
            "n_tied": dim_tied,
        }
        n_comparisons += dim_comparisons
        n_tied += dim_tied
        for name in dim_patterns:
            if name not in ordered:
                ordered.append(name)
            wins[name] = wins.get(name, 0) + dim_wins.get(name, 0)

    summary = {
        "wins": {name: wins.get(name, 0) for name in ordered},
        "n_comparisons": n_comparisons,
        "n_tied": n_tied,
        "by_dim": by_dim,
    }

    if print_stats:
        print("Best-strategy counts (raw loss, all trials × resolutions × dimensions):")
        print("  comparisons: {}, ties (shared min): {}".format(n_comparisons, n_tied))
        for name in ordered:
            count = wins.get(name, 0)
            pct = 100.0 * count / n_comparisons if n_comparisons else 0.0
            print("  {}: {} ({:.2f}%)".format(name, count, pct))
        print()

    if csv_path:
        if not os.path.isabs(csv_path) and 2 in run_dirs:
            csv_path = os.path.join(run_dirs[2], csv_path)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Pattern", "N best", "Percent of comparisons"])
            for name in ordered:
                count = wins.get(name, 0)
                pct = 100.0 * count / n_comparisons if n_comparisons else 0.0
                writer.writerow([name, count, "{:.6g}".format(pct)])
            writer.writerow([])
            writer.writerow(["n_comparisons", n_comparisons, ""])
            writer.writerow(["n_tied", n_tied, ""])
        print("Saved best-strategy counts to {}".format(csv_path))
        print()

    return summary


DIM_WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}


def _mean_pct_improvement_over_n(means, strategy, reference):
    """Unweighted mean over resolutions of 100 * (reference − strategy) / reference.

    Positive means the strategy has lower loss than the reference (same
    convention as the old simple-function manuscript text). Resolutions
    with a zero or non-finite reference mean are skipped.
    """
    if strategy not in means or reference not in means:
        return None
    svals = means[strategy]
    rvals = means[reference]
    parts = []
    for s, r in zip(svals, rvals):
        if not np.isfinite(s) or not np.isfinite(r) or r == 0:
            continue
        parts.append(100.0 * (r - s) / r)
    if not parts:
        return None
    return float(np.mean(parts))


def _format_pct_latex(value):
    if value is None or not np.isfinite(value):
        return r"n/a"
    return r"${:.0f}\%$".format(int(round(value)))


def print_averaged_n_improvements(
    run_dirs,
    dimensions=(2, 3, 4, 5),
    patterns=None,
    strategy=None,
    vs=("grid", "random"),
    hyperuniform_reference=None,
):
    """Print mean-over-N percent improvements, plus a pasteable LaTeX sentence.

    Uses raw per-resolution means (not the figure's relative-loss metric).
    ``strategy`` defaults to the hyperuniform reference for each dimension.
    """
    if isinstance(run_dirs, str):
        run_dirs = {dimensions[0]: run_dirs}

    vs = tuple(vs)
    dims_used = [d for d in dimensions if d in run_dirs]
    by_vs = {ref: [] for ref in vs}
    strategy_names = []

    for dim in dims_used:
        dim_patterns = _patterns_for_dimension(patterns, dim)
        results = load_run_results(run_dirs[dim], dim)
        if results is None:
            for ref in vs:
                by_vs[ref].append(None)
            strategy_names.append(strategy or _hyperuniform_reference_for_dim(
                dim, hyperuniform_reference, dim_patterns
            ))
            continue
        means, _, _ = _means_stderr_from_results(results, dim_patterns)
        strat = strategy or _hyperuniform_reference_for_dim(
            dim, hyperuniform_reference, dim_patterns
        )
        strategy_names.append(strat)
        for ref in vs:
            by_vs[ref].append(_mean_pct_improvement_over_n(means, strat, ref))

    print("Average improvement over all considered N (positive = lower loss):")
    if strategy_names and len(set(strategy_names)) == 1:
        print("  strategy: {}".format(strategy_names[0]))
    else:
        print("  strategy per dimension: {}".format(
            {d: n for d, n in zip(dims_used, strategy_names)}
        ))
    for ref in vs:
        bits = []
        for dim, val in zip(dims_used, by_vs[ref]):
            if val is None:
                bits.append("{}D n/a".format(dim))
            else:
                bits.append("{}D {:.2f}%".format(dim, val))
        print("  vs {}: {}".format(ref, ", ".join(bits)))

    dim_phrase = ", ".join(DIM_WORDS.get(d, str(d)) for d in dims_used)
    if len(dims_used) > 1:
        *rest, last = [DIM_WORDS.get(d, str(d)) for d in dims_used]
        dim_phrase = ", ".join(rest) + ", and " + last

    grid_tex = ", ".join(_format_pct_latex(v) for v in by_vs.get("grid", []))
    rand_tex = ", ".join(_format_pct_latex(v) for v in by_vs.get("random", []))
    print(
        "When averaged over results for all considered $N$, hyperuniform "
        "patterns provide improvements over grid search of {} and "
        "improvements over random search of {} in {} dimensions, "
        "respectively.".format(grid_tex, rand_tex, dim_phrase)
    )
    print()
    return {ref: dict(zip(dims_used, by_vs[ref])) for ref in vs}


def plot_simple_function_comparison(
    run_dir,
    d,
    ax,
    patterns,
    colors=None,
    labels=None,
    baseline="grid",
    position="up",
    y_lim=None,
    print_stats=True,
    hyperuniform_reference=None,
    wilcoxon_patterns=None,
    loss_offset_per_dim=0,
):
    """
    Plot loss vs. number of points for selected patterns from one run directory.

    Parameters
    ----------
    run_dir : str
        Path to run_YYYYMMDD_HHMMSS (or folder containing results_d_{d}.pkl).
    d : int
        Dimensionality (must match the run).
    ax : matplotlib.axes.Axes
    patterns : list of str or dict[int, list of str]
        Pattern names for this dimension, or a per-dimension mapping (use the
        latter when plotting a single panel from the same PATTERNS dict as the
        multi-panel figure).
    colors : dict or list, optional
        Pattern name -> color, or list aligned with patterns. Defaults used if missing.
    labels : dict or list, optional
        Legend labels.
    baseline : str
        Pattern used in (strategy − baseline) / |baseline| (y=0). Default "grid".
    position : str
        "up" or "down" for x-axis tick placement (multi-panel figures).
    y_lim : tuple, optional
        (ymin, ymax). Defaults to ylim_from_data(...) over plotted points.
    print_stats : bool
        Print summary statistics to stdout.
    wilcoxon_patterns : list of str or dict[int, list of str], optional
        Patterns to include in the Wilcoxon test. Defaults to ``patterns``.
        The Wilcoxon reference is added automatically if missing.
    loss_offset_per_dim : float
        Add loss_offset_per_dim * d to every pattern mean before plotting
        (e.g. 40 for Styblinski-Tang so grid-at-origin losses stay positive).

    Returns
    -------
    dict
        means, stderr, numbers, rel_means, rel_stderr, metadata
    """
    patterns = _patterns_for_dimension(patterns, d)
    results = load_run_results(run_dir, d)
    if results is None:
        print("Warning: no results file for {}D in {}; leaving empty panel.".format(d, run_dir))
        _style_empty_panel(ax, d, position=position, y_lim=y_lim)
        return _empty_plot_summary()

    metadata = results.get("metadata", {})
    available = set(metadata.get("pattern_names", []))
    for block in results["everything_scores"]:
        available.update(block.keys())

    missing = [p for p in patterns if p not in available]
    if missing:
        raise ValueError(
            "Patterns not in run {}: {}. Available: {}".format(run_dir, missing, sorted(available))
        )
    if baseline not in patterns:
        raise ValueError("baseline '{}' must be included in patterns".format(baseline))

    color_map, marker_map, label_map = _resolve_styles(patterns, colors, labels)
    means, stderr, numbers = _means_stderr_from_results(results, patterns)
    means = _apply_loss_offset(means, d, loss_offset_per_dim)
    rel_means, rel_stderr = _relative_to_baseline(means, stderr, baseline)
    largest_idx = _largest_resolution_index(results)
    wilcoxon_ref = _hyperuniform_reference_for_dim(d, hyperuniform_reference, patterns)
    wilcoxon_names = _wilcoxon_patterns_for_dimension(
        wilcoxon_patterns, patterns, d, wilcoxon_ref
    )
    missing_w = [p for p in wilcoxon_names if p not in available]
    if missing_w:
        raise ValueError(
            "Wilcoxon patterns not in run {}: {}. Available: {}".format(
                run_dir, missing_w, sorted(available)
            )
        )
    wilcoxon_summary = _paired_wilcoxon_vs_reference(results, wilcoxon_names, wilcoxon_ref)

    if print_stats:
        _print_summary(
            d,
            patterns,
            means,
            rel_means,
            largest_idx=largest_idx,
            wilcoxon_summary=wilcoxon_summary,
            baseline=baseline,
        )

    for name in patterns:
        x = numbers[name]
        y = rel_means[name]
        ye = rel_stderr[name]
        color = color_map[name]
        marker = marker_map[name]
        ax.errorbar(
            x, y, yerr=ye, color=color, elinewidth=3, capthick=3, linestyle="none"
        )
        ax.plot(
            x,
            y,
            marker=marker,
            markersize=15,
            linestyle="-",
            linewidth=3,
            color=color,
            label=label_map[name],
        )
        ax.fill_between(
            x,
            [y[j] - ye[j] for j in range(len(y))],
            [y[j] + ye[j] for j in range(len(y))],
            color=color,
            alpha=0.1,
        )

    x_ref = numbers.get("RSA", numbers[baseline])
    if not x_ref:
        x_ref = numbers[patterns[0]]
    x_min = min(x_ref) * 0.9
    x_max = max(x_ref) * 1.1
    N = int(round(x_max ** (1.0 / d))) if d > 0 else 1

    ax.set_title("{}D".format(d), fontsize=30, y=0.84, x=0.1)
    ax.set_xscale("log")
    ax.set_xlim((max(x_min, 2**d * 0.9 if d <= 5 else x_min), x_max))

    if y_lim is None:
        y_lim = ylim_from_data(rel_means, stderr=rel_stderr, d=d)
    ax.set_ylim(y_lim)

    if d == 2:
        maks_power = int(np.log10(400 * 1.1))
    else:
        maks_power = int(np.log10(max(x_max, 1.1)))

    values_upper = []
    ticks_upper = []
    for i in range(1, maks_power + 1):
        tick_val = 10**i
        if tick_val >= x_min and tick_val < x_max:
            values_upper.append(tick_val)
            ticks_upper.append("$10^{}$".format(i))

    if values_upper:
        ax.set_xticks(values_upper, ticks_upper, fontsize=25)

    ax.tick_params(labelsize=25)
    if position == "up":
        ax.tick_params(
            "x",
            bottom=True,
            top=True,
            labelbottom=False,
            labeltop=True,
            labelsize=25,
        )

    ax.yaxis.set_label_coords(1.05, 0.5)
    ax.set_axisbelow(True)
    ax.set_facecolor("whitesmoke")

    return {
        "means": means,
        "stderr": stderr,
        "numbers": numbers,
        "rel_means": rel_means,
        "rel_stderr": rel_stderr,
        "largest_idx": largest_idx,
        "wilcoxon_summary": wilcoxon_summary,
        "wilcoxon_reference": wilcoxon_ref,
        "wilcoxon_patterns": wilcoxon_names,
        "metadata": metadata,
    }


def plot_comparison_figure(
    run_dirs,
    dimensions=(2, 3, 4, 5),
    patterns=None,
    colors=None,
    labels=None,
    baseline="grid",
    save_path="Comparison_simple.jpg",
    show=True,
    print_stats=True,
    summary_csv_path=None,
    hyperuniform_reference=None,
    wilcoxon_patterns=None,
    func_name="simple",
    loss_offset_per_dim=0,
    make_figure=True,
):
    """
    2x2 panel figure: one run directory per dimension.

    Parameters
    ----------
    run_dirs : dict[int, str] or str
        Mapping dimension -> run folder, or a single folder (same run only works for one d).
    patterns : list[str] or dict[int, list[str]], optional
        Shared pattern list for all dimensions, or per-dimension pattern lists.
    hyperuniform_reference : str or dict[int, str], optional
        Wilcoxon reference hyperuniform pattern per dimension. Defaults to hyperuniform_0.4.
    wilcoxon_patterns : list[str] or dict[int, list[str]], optional
        Patterns to test with Wilcoxon. Defaults to ``patterns``.
    loss_offset_per_dim : float
        Passed to plot_simple_function_comparison; see that function.
    make_figure : bool
        If False, skip saving/showing the figure and only write the CSV table
        (set ``summary_csv_path``). Default True.
    """
    if isinstance(run_dirs, str):
        run_dirs = {dimensions[0]: run_dirs}
    first_dir = next((run_dirs[d] for d in dimensions if d in run_dirs), os.getcwd())
    save_path = os.path.join(first_dir, save_path) if not os.path.isabs(save_path) else save_path
    summary_csv_path = os.path.join(first_dir, summary_csv_path) if summary_csv_path and not os.path.isabs(summary_csv_path) else summary_csv_path

    fig, axs = plt.subplots(2, 2, figsize=(7 * 3, 5 * 3))
    positions = {(2, 0): "up", (3, 1): "up", (4, 0): "down", (5, 1): "down"}
    panel_map = {2: (0, 0), 3: (0, 1), 4: (1, 0), 5: (1, 1)}
    summaries_by_dim = {}

    for dim in dimensions:
        row, col = panel_map[dim]
        pos = positions.get((dim, col if col == 1 else row), "down")
        if dim not in run_dirs:
            print("Warning: no run directory for {}D; leaving empty panel.".format(dim))
            _style_empty_panel(axs[row, col], dim, pos)
            continue
        dim_patterns = _patterns_for_dimension(patterns, dim)
        summaries_by_dim[dim] = plot_simple_function_comparison(
            run_dirs[dim],
            dim,
            axs[row, col],
            patterns=dim_patterns,
            colors=colors,
            labels=labels,
            baseline=baseline,
            position=pos,
            print_stats=False,
            hyperuniform_reference=hyperuniform_reference,
            wilcoxon_patterns=wilcoxon_patterns,
            loss_offset_per_dim=loss_offset_per_dim,
        )

    if print_stats and summaries_by_dim:
        _print_summary_table(
            patterns=patterns,
            dimensions=dimensions,
            summaries_by_dim=summaries_by_dim,
            csv_path=summary_csv_path,
            hyperuniform_reference=hyperuniform_reference,
        )

    if not make_figure:
        plt.close(fig)
        return None

    axs[0, 0].tick_params(
        "y", left=True, right=True, labelleft=True, labelright=False, labelsize=25
    )
    axs[0, 1].tick_params(
        "y", left=True, right=True, labelleft=False, labelright=True, labelsize=25
    )
    axs[1, 0].tick_params(
        "y", left=True, right=True, labelleft=True, labelright=False, labelsize=25
    )
    axs[1, 1].tick_params(
        "y", left=True, right=True, labelleft=False, labelright=True, labelsize=25
    )

    legend_entries = {}
    for ax in axs.flat:
        handles, legend_labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, legend_labels):
            if label not in legend_entries:
                legend_entries[label] = handle

    # legend_entries = [] # [TODO] change this

    if legend_entries:
        legend_row, legend_loc = _func_legend_location(func_name)
        leg = axs[legend_row, 1].legend(
            legend_entries.values(),
            legend_entries.keys(),
            # fontsize=25,
            fontsize=15,
            markerscale=1.2,
            loc=legend_loc,
        )
        for line in leg.get_lines():
            line.set_linewidth(3)

    fig.suptitle(
        f"Comparison for the {_func_display_name(func_name)} function in 2-5 dimensions", fontsize=35
    )
    label_map = dict(DEFAULT_LABELS)
    if isinstance(labels, dict):
        label_map.update(labels)
    baseline_label = _usual_pattern_label(baseline, label_map).lower()
    fig.text(
        -0.02,
        0.5,
        "(Loss of strategy − loss of {}) / |loss of {}|".format(
            baseline_label, baseline_label
        ),
        va="center",
        rotation="vertical",
        fontsize=30,
    )
    fig.text(
        0.5,
        -0.02,
        "Number of sampled points",
        ha="center",
        fontsize=30,
    )

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=600)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


if __name__ == "__main__":
    # Example: set one run directory per dimension after running Generic_generate_results.py
    SHOW = True

    PATTERNS = {
        2: [
            "grid",
            "random",
            "gridrandom",
            "latin",
            "sobol",
            "halton",
            "hyperuniform_0.4",
            "hyperuniform_0.3"
        ],
        3: ["grid", "random", "gridrandom", "latin", "sobol", "halton", "hyperuniform_0.3", "hyperuniform_0.4"],
        4: ["grid", "random", "gridrandom", "latin", "sobol", "halton", "hyperuniform_0.3", "hyperuniform_0.4"],
        5: ["grid", "random", "gridrandom", "latin", "sobol", "halton", "hyperuniform_0.3", "hyperuniform_0.4"],
    }

    # Patterns used for the simple function figure in the main manuscript
    # PATTERNS = {
    #     2: ["grid", "random", "hyperuniform_0.4"],
    #     3: ["grid", "random", "hyperuniform_0.4"],
    #     4: ["grid", "random", "hyperuniform_0.4"],
    #     5: ["grid", "random", "hyperuniform_0.4"],
    # }


    COLORS = {
        "grid": "green",
        "random": "blue",
        "gridrandom": "black",
        "latin": "teal",
        "sobol": "navy",
        "halton": "olive",
        "hyperuniform_0.49": "red",
        "hyperuniform_0.38": "crimson",
        "hyperuniform_0.25": "magenta",
        "hyperuniform_0.13": "violet",
        "hyperuniform_0.4": "red",
        "hyperuniform_0.3": "violet",
    }

    dir_res = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    # These are the patterns that will be plotted.
    names = ["simple"]
    # names = ["langermann", "styblinski_tang"]
    # names = ["branin", "rastrigin", "schwefel", "zakharov"]
    # names = ["branin_rotated", "langermann_rotated", "rastrigin_rotated", "schwefel_rotated"]
    # names = ["langermann_rotated"]

    for name in names:
        dir_results = os.path.join(dir_res, name)


        plot_comparison_figure(
            run_dirs = {2: dir_results, 3: dir_results, 4: dir_results, 5: dir_results},
            dimensions=(2, 3, 4, 5),
            save_path=f"Comparison_{name}.jpg",
            patterns=PATTERNS,
            # wilcoxon_patterns=["random", "latin", "sobol"],  # optional; defaults to patterns
            # hyperuniform_reference="hyperuniform_0.4",       # who Wilcoxon tests against
            colors=COLORS,
            baseline="random",
            # baseline="grid",
            summary_csv_path=f"Comparison_{name}_summary.csv",
            func_name=name,
            loss_offset_per_dim=0,
            show=SHOW,
            # make_figure=False,  # CSV only; no jpg / window
        )

        count_best_strategy_trials(
            run_dirs={2: dir_results, 3: dir_results, 4: dir_results, 5: dir_results},
            dimensions=(2, 3, 4, 5),
            patterns=PATTERNS,
            csv_path=f"Comparison_{name}_best_counts.csv",
        )

        print_averaged_n_improvements(
            run_dirs={2: dir_results, 3: dir_results, 4: dir_results, 5: dir_results},
            dimensions=(2, 3, 4, 5),
            patterns=PATTERNS,
        )

