from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
STEALTHY_ROOT = BASE_DIR.parent / "stealthySmallHyperP"

# Parent resolution used when carving from a higher-resolution pattern.
SOURCE_2D = 400
SOURCE_SIDE = 5


def _stealth_label(stealth):
    return format(float(stealth), "g")


def _trial_directory(d, number, stealth):
    return STEALTHY_ROOT / str(d) / str(number) / _stealth_label(stealth)


def _is_trial_file(path):
    """True for numbered trial patterns (e.g. 0.dat); excludes kvectors.dat."""
    return path.suffix == ".dat" and path.stem.isdigit()


def _available_trial_count(d, number, stealth):
    directory = _trial_directory(d, number, stealth)
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.iterdir() if _is_trial_file(path))


def _resolve_trial_number(d, number, stealth, trial_number):
    """Map out-of-range trial indices onto available files via modulo."""
    n = _available_trial_count(d, number, stealth)
    if n > 0 and trial_number >= n:
        return trial_number % n
    return trial_number


def stealthy_pattern_path(d, number, stealth, trial_number):
    """Path to one precomputed pattern: stealthySmallHyperP/{d}/{number}/{stealth}/{trial}.dat"""
    return _trial_directory(d, number, stealth) / f"{int(trial_number)}.dat"


def load_stealthy_pattern(d, number, stealth, trial_number):
    """Load raw coordinates from a stealthySmallHyperP .dat file (in [0, 1]^d)."""
    path = stealthy_pattern_path(d, number, stealth, trial_number)
    if not path.is_file():
        raise FileNotFoundError(
            "No stealthy pattern at {} (d={}, number={}, stealth={}, trial={})".format(
                path, d, number, stealth, trial_number
            )
        )

    points = np.loadtxt(path)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] != d:
        raise ValueError(
            "Expected {} columns in {}, got {}".format(d, path, points.shape[1])
        )
    return points


def _default_source_number(d):
    """Highest-resolution parent pattern count for carving in dimension ``d``."""
    if d == 2:
        return SOURCE_2D
    return SOURCE_SIDE ** d


def _carve_subpattern(parent_points, source_number, number, k, d, trial_number):
    """
    Carve a random axis-aligned window from a parent pattern in [0, 1]^d.

    Window side length is chosen so the expected number of points inside matches
    ``number``, using parent resolution ``source_number`` over unit volume.
    """
    side = (number / source_number) ** (1.0 / d)
    init_length = 1.0
    rng = np.random.default_rng(trial_number)

    low = 0.05 * init_length + side / 2
    high = 0.95 * init_length - side / 2
    if low <= high:
        center = [rng.uniform(low, high) for _ in range(d)]
    else:
        center = [0.5 * init_length] * d

    carved = [
        list(p)
        for p in parent_points
        if all(
            p[i] - center[i] < side / 2 and p[i] - center[i] > -side / 2 for i in range(d)
        )
    ]

    new_min = [-k / 2 for _ in range(d)]
    new_max = [k / 2 for _ in range(d)]
    old_min = [center[i] - side / 2 for i in range(d)]
    old_max = [center[i] + side / 2 for i in range(d)]

    from Generic_generate_results import transform_tiling

    transform_tiling(carved, old_min, old_max, new_min, new_max, d)
    return carved


def make_hyperuniform(number, k, d, stealth=0.49, trial_number=None, seed=None):
    """
    Load a stealthy hyperuniform point pattern from stealthySmallHyperP.

    Files live at stealthySmallHyperP/{d}/{number}/{stealth}/{trial_number}.dat
    with one point per line and coordinates in [0, 1]^d.

    If no file exists at the requested resolution, the pattern is carved from
    the highest-resolution parent (400 points in 2D, 5^d otherwise) with the
    same stealth and trial_number.

    trial_number selects the configuration (same index as run_pattern_comparison
    trials). If omitted, seed is used instead (default 0). When trial_number
    exceeds the number of available files, it is reduced modulo that count.

    Returns points in [-k/2, k/2]^d as a list of coordinate vectors.
    """
    if trial_number is None:
        trial_number = 0 if seed is None else int(seed)

    resolved_trial = _resolve_trial_number(d, number, stealth, trial_number)

    if not stealthy_pattern_path(d, number, stealth, resolved_trial).is_file():
        source_number = _default_source_number(d)
        parent_trial = _resolve_trial_number(
            d, source_number, stealth, trial_number
        )
        parent = load_stealthy_pattern(
            d, source_number, stealth, parent_trial
        )
        return _carve_subpattern(
            parent, source_number, number, k, d, trial_number
        )

    points = load_stealthy_pattern(d, number, stealth, resolved_trial)
    scaled = points * k - k / 2
    return scaled.tolist()
