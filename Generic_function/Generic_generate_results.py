#This program generates results for the generic function and saves them into a file

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os
from datetime import datetime
from itertools import combinations, product
from mpl_toolkits import mplot3d
import pickle
import argparse
from functools import partial
from scipy.stats import qmc
# from Hyperuniform import make_hyperuniform
from Hyperuniform import make_hyperuniform
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

random.seed(22)
np.random.seed(22)

################generic function used in the comparison

def f(x, point, A, w):
    #x - the vector
    #A - the matrix making a linear transformation of the coordinates
    #w - the vector of weights
    #p - the point from which we calculate the distance

    return np.linalg.norm(np.dot([x[i] - point[i] for i in range(d)], A)*w)**2

################transforming a tiling to required maximum/minimum of hyperparameter space

def transform_tiling(tiling, old_min, old_max, new_min, new_max, d):
    #d is the number of dimensions
    #old_min is the vector with the old minimum, old_max - maximum
    #new_min is the new minimum, new_max - maximum
    for i in range(d):
        a = (new_max[i]-new_min[i])/(old_max[i]-old_min[i])
        b = (new_min[i]*old_max[i]-old_min[i]*new_max[i])/(old_max[i]-old_min[i])
        for point in tiling:
            point[i] = a*point[i]+b

################making a hyperuniform point pattern

def get_hyper(d, directory):
    #k - the sidelength of the square
    #number - the number if points
    #d - the number of dimensions


    x = []
    with open(directory) as f:
        line = f.readline()
        while line:
            x.append([float(i) for i in line.split()])
            x[-1] = x[-1][:d]
            line = f.readline()

    density = 0 #the density

    for p in x:
        if all([p[i]>=0.05 and p[i]<=0.95 for i in range(d)]):
            density += 1


    density = density/(0.9**d)

    return x, density


################ Sobol

def make_sobol(number, k, d, trial_number=None, seed=None):
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    points = sampler.random(number)

    return qmc.scale(points, [-k/2]*d, [k/2]*d).tolist()

################ Halton

def make_halton(number, k, d, trial_number=None, seed=None):
    sampler = qmc.Halton(d=d, scramble=True, seed=seed)
    points = sampler.random(number)

    return qmc.scale(points, [-k/2]*d, [k/2]*d).tolist()

################making a grid with a given number of points

def make_grid(number, k, d, trial_number=None):
 #number - the intended number of points   
 #k - the length of the square
 #d - the number of dimensions
  gr = pow(number, 1/d)
  if math.ceil(gr)**d == number:
    gr = math.ceil(gr)
  else:
    gr = int(gr)
  g = np.linspace(-k/2, k/2, gr)
  listgrid = []
  for c in product(g, repeat = d):
       listgrid.append(list(c))
  return listgrid

################making a random pattern with a given number of points

def make_random(number, k, d, trial_number=None):
    #number - the intended number of points
    #k - the side length of the squares
    #d - the number of dimensions
    listrand = []
    for r in range(number):
      listrand.append([random.uniform(-k/2, k/2) for j in range(d)])
    return listrand

################making a gridrandom pattern with a given number of points

def make_gridrandom(number, k, d, trial_number=None):
    #number - the intended number of points
    #k - the length of the squares
    #d - the number of dimensions
    listgridrand = []
    gr = pow(number, 1/d)
    if math.ceil(gr)**d == number:
      gr = math.ceil(gr)
    else:
      gr = int(gr)
    g = np.linspace(-k/2, k/2, gr+1)
    for c in product(g[:gr], repeat=d):
          listgridrand.append([c[j]+random.uniform(0, k/(gr)) for j in range(d)])
    return listgridrand

################checking if a point lies on the square

def if_square(v, d, k):
  #v - the vector
  #d - the number of dimensions of v
  #k - length of the square
  for i in range(d):
    if v[i]>k/2 or v[i]<-k/2:
      return False
  return True

################making the RSA pattern

def make_RSA(number, Length, d, trial_number=None):
    #number - expected number of points 
    #Length - side length of the square
    #d - number of dimensions

    ##saturation density
    def density_RSA(d):
      if d==1:
        return 0.7475
      if d==2:
        return 0.5470735
      if d==3:
        return 0.3841307
      if d==4:
        return 0.2600781
      if d==5:
        return 0.1707761
      if d==6:
        return 0.109302
      if d==7:
        return 0.068404
      if d==8:
        return 0.0423
      return 1/2**(d)

    ##volume of a d-dimensional unit ball
    def ball(d):
      if d==0:
        return 1
      if d==1:
        return 2
      return 2*np.pi*ball(d-2)/d

    D0 = 2*Length*(density_RSA(d)/(number*ball(d)))**(1/d) #diameter of the spheres
    if d==2 or d==3:
      D = D0/(1-D0/(1.5*Length))
    if d==4:
      D = D0/(1-D0/(1.65*Length))
    if d>=5 and d<=7:
      D = D0/(1-D0/(2*Length))
    if d==8:
      D = D0/(1-D0/(Length))

    if D>Length:
      D=D0
    
    bigvoxel_amount = int(Length//D)
    bigvoxel = Length/(Length//D) #bigvoxel size
    v = Length/(Length//(D)) #smallvoxel initial sizes
    N = int(Length//(D)) #initial amount of smallvoxels

    ##for the neighborhood list

    n_shape = [bigvoxel_amount for i in range(d)]
    n = np.empty(n_shape, dtype = object)
    n.fill([])
  
    ##checking whether a sphere with center c can be added

    def check_sphere(c):
      c1 = c/bigvoxel
      c1 = c1.astype(int)
      for t in product([0, -1, 1], repeat = d):
        try:
            for a in n[tuple(c1 + np.array(t))]:
              if np.linalg.norm(a - c) < D:
                return False
        except IndexError:
            continue
      n[tuple(c1)].append(c)
      return True

    ##checking whether a voxel with corner c and length v can be discarded

    def check_voxel(c, vec0, vec0_dist):
      c1 = c/bigvoxel
      c1 = c1.astype(int)
      for t in product([0, -1, 1], repeat = d):
        try:
            for a0 in n[tuple(c1 + np.array(t))]:
              if np.linalg.norm(a0-vec0-c) < D-vec0_dist:
                return False
        except IndexError:
            continue
      return True

    count = 0

    list_RSA = []

    #maximum number od trials after which we proceed to the next step
    count_MAX = 30

    #vector from corner to center of a voxel
    vec = np.array([v/2 for i in range(d)])
    vec_dist = np.linalg.norm(vec)

    vlist = []

    ##phase 1 - searching through all space
    while count < count_MAX:
      c = np.array([random.uniform(0, Length) for i in range(d)])
      if check_sphere(c):
        list_RSA.append(c)
        count=0
      else:
        count+=1

    print("Phase 1 complete")

    ##setup for the second phase - creating the voxel list
    l = [i*v for i in range(N)]
    for t in product(l, repeat = d):
      if check_voxel(np.array(t), vec, vec_dist):
        vlist.append(np.array(t))
    vnum = len(vlist)

    ##phase 2 - iterating until the voxel list is empty 

    while len(vlist) >= 1:
      count=0
      while count < count_MAX:
        i = int(random.uniform(0, vnum))
        c = np.array([random.uniform(0, v) for j in range(d)]) + vlist[i]
        if check_sphere(c):
          count=0
          list_RSA.append(c)
        else:
          count+=1
      print("Current value of 2^(d) * (length of smallvoxel list)", 2**d * len(vlist))

      if d>=4 and d<=4 and 2**d * len(vlist) > 100000:
        vlist1 = []
        for voxel in vlist:
          if check_voxel(voxel, vec, vec_dist):
            vlist1.append(voxel)
        vlist = list(vlist1)
              
        print("Moving to phase 3")
        break

      if 5<=d and 2**d * len(vlist) > 10000:
        vlist1 = []
        for voxel in vlist:
          if check_voxel(voxel, vec, vec_dist):
            vlist1.append(voxel)
        vlist = list(vlist1)
              
        print("Moving to phase 3")
        break
      
      v, vec, vec_dist = v/2, vec/2, vec_dist/2

      vlist1 = []

      for voxel in vlist:
        if check_voxel(voxel, 2*vec, 2*vec_dist):
          for t in product([0, v], repeat=d):
            if check_voxel(voxel+np.array(t), vec, vec_dist):
              vlist1.append(voxel+np.array(t))
      vlist=list(vlist1)
      vnum=len(vlist)

    if len(vlist)>=1:
      count_max = max(int(500000/len(vlist)), 30)
    
    random.shuffle(vlist)

    ##phase 3

    while len(vlist)>=1:
        count = 0
        while count<=count_max:
          c = np.array([random.uniform(0, v) for j in range(d)]) + vlist[-1]
          if check_sphere(c):
            count=0
            list_RSA.append(c)
          else:
            count+=1
        vlist.pop()
    
    return list_RSA


################generating the Latin Hypercube Sampling pattern with a given number of points

def make_latin(number, k, d, trial_number=None):
    #number - intended number of points
    #k - the side length of the square
    #d - the number of dimensions
    
    l = [i for i in range(number)]
    rng = np.random.default_rng()
    m = [1 for i in range(d)]
    m[d-1]=l
    for i in range(d-1):
        m[i] = rng.permutation(l)
    listlatin = []
    for i in range(number):
        listlatin.append([k*(m[j][l[i]])/(number-1)-(k/2) for j in range(d)])
    return listlatin

################ Rastrigin function

def rastrigin_point(x):
    d = len(x)

    total = 10 * d

    for xi in x:
        total += xi * xi - 10 * math.cos(2 * math.pi * xi)

    return total


def loss_rastrigin(point_pattern,
                  dimensions,
                  number_of_points,
                  trial_number,
                  k=1,
                  dim_reduction=lambda x: x):
    # dim_reduction give the number of dimensions that matter
    scale = 10.24 / k

    rng = random.Random(trial_number)

    # tiny random shift in canonical Rastrigin coordinates
    shift = [
        rng.uniform(-1, 1)
        for _ in range(dimensions)
    ]

    best_value = float("inf")

    for point in point_pattern:

        # rescale from [-k/2, k/2]
        # to canonical [-5.12, 5.12]
        transformed = [
            scale * point[j] + shift[j]
            for j in range(dim_reduction(dimensions))
        ]

        value = rastrigin_point(transformed)

        if value < best_value:
            best_value = value

    return best_value

################ Styblinski-Tang function

def styblinski_tang_point(x):
    total = 0

    for xi in x:
        total += xi**4 - 16 * xi**2 + 5 * xi

    return 0.5 * total


def loss_styblinski_tang(point_pattern,
                         dimensions,
                         number_of_points,
                         trial_number,
                         k=1,
                         dim_reduction=lambda x: x):
    # canonical domain is [-5, 5]^d
    scale = 10 / k

    best_value = float("inf")

    for point in point_pattern:
        # rescale from [-k/2, k/2] to [-5, 5]
        transformed = [
            scale * point[j]
            for j in range(dim_reduction(dimensions))
        ]

        value = styblinski_tang_point(transformed)

        if value < best_value:
            best_value = value

    return best_value

################ Langermann function

def langermann_point(x):
    # Standard d=2 parameters from SFU
    c = [1, 2, 5, 2, 3]
    A = [
        [3, 5],
        [5, 2],
        [2, 1],
        [1, 4],
        [7, 9],
    ]

    total = 0

    for i in range(5):
        dist2 = 0
        for j in range(len(x)):
            dist2 += (x[j] - A[i][j])**2

        total += c[i] * math.exp(-dist2 / math.pi) * math.cos(math.pi * dist2)

    return -total


def loss_langermann(point_pattern,
                    dimensions,
                    number_of_points,
                    trial_number,
                    k=1,
                    dim_reduction=lambda x: x):
    # canonical domain is [0, 10]^2
    scale = 10 / k

    best_value = float("inf")

    for point in point_pattern:
        # rescale from [-k/2, k/2] to [0, 10]
        transformed = [
            scale * (point[j] + k / 2)
            for j in range(2)
        ]

        value = langermann_point(transformed)

        if value < best_value:
            best_value = value

    return best_value

################ Zakharov function

def zakharov_point(x):
    sum1 = 0
    sum2 = 0

    for i, xi in enumerate(x, start=1):
        sum1 += xi * xi
        sum2 += 0.5 * i * xi

    return sum1 + sum2**2 + sum2**4


def loss_zakharov(point_pattern,
                  dimensions,
                  number_of_points,
                  trial_number,
                  k=1,
                  dim_reduction=lambda x: x):
    # canonical domain is usually [-5, 10]^d
    scale = 15 / k

    rng = random.Random(trial_number)

    # shift away from origin in canonical Zakharov coordinates
    shift = [
        rng.uniform(-4, 4)
        for _ in range(dim_reduction(dimensions))
    ]

    best_value = float("inf")

    for point in point_pattern:
        # rescale from [-k/2, k/2] to [-5, 10]
        transformed = [
            scale * (point[j] + k / 2) - 5 - shift[j]
            for j in range(dim_reduction(dimensions))
        ]

        value = zakharov_point(transformed)

        if value < best_value:
            best_value = value

    return best_value


################ Schwefel function

def schwefel_point(x):
    d = len(x)

    total = 418.9829 * d

    for xi in x:
        total -= xi * math.sin(math.sqrt(abs(xi)))

    return total


def loss_schwefel(point_pattern,
                  dimensions,
                  number_of_points,
                  trial_number,
                  k=1,
                  dim_reduction=lambda x: x):
    # canonical domain is [-500, 500]^d
    scale = 1000 / k

    best_value = float("inf")

    for point in point_pattern:
        # rescale from [-k/2, k/2] to [-500, 500]
        transformed = [
            scale * point[j]
            for j in range(dim_reduction(dimensions))
        ]

        value = schwefel_point(transformed)

        if value < best_value:
            best_value = value

    return best_value

################ The Branin loss function here

def branin(x1, x2):

    a = 1.0
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)

    return (
        a * (x2 - b * x1**2 + c * x1 - r)**2
        + s * (1 - t) * math.cos(x1)
        + s
    )


def loss_branin(point_pattern,
                  dimensions,
                  number_of_points,
                  trial_number,
                  k, 
                  dim_reduction=lambda x: 2):

    if dim_reduction(dimensions) != 2:
        raise ValueError("Branin function is only defined in 2 dimensions")

    best_value = float("inf")

    for point in point_pattern:

        px, py = point[0], point[1]

        # [-k/2, k/2] -> [-5, 10]
        x1 = ((px + k / 2.0) / k) * 15.0 - 5.0

        # [-k/2, k/2] -> [0, 15]
        x2 = ((py + k / 2.0) / k) * 15.0

        value = branin(x1, x2)

        if value < best_value:
            best_value = value

    return best_value

################ The generic (random quadratic) loss function

_generic_trial_cache = {}


def loss_generic(point_pattern, dimensions, number_of_points, trial_number, k=1):
    key = (dimensions, number_of_points, trial_number)
    if key not in _generic_trial_cache:
        A = np.array(
            [[np.random.uniform(-1, 1) for _ in range(dimensions)] for _ in range(dimensions)]
        )
        A = A / np.linalg.norm(A)
        w = np.array([np.random.uniform(0, 1000) for _ in range(dimensions)])
        w = 100 * w / np.linalg.norm(w)
        point = [np.random.uniform(-k / 2, k / 2) for _ in range(dimensions)]
        _generic_trial_cache[key] = {"A": A, "w": w, "point": point}

    trial_state = _generic_trial_cache[key]
    point = trial_state["point"]
    A = trial_state["A"]
    w = trial_state["w"]
    dim = len(point)

    def objective(x):
        return np.linalg.norm(
            np.dot([x[i] - point[i] for i in range(dim)], A) * w
        ) ** 2

    best = objective(point_pattern[0])
    for x in point_pattern:
        val = objective(x)
        if val < best:
            best = val
    return best


loss_generic.trial_cache = _generic_trial_cache

################ The quadratic trial 

def _sample_random_quadratic_trial(dimensions, k):
    """Draw one random quadratic benchmark instance (A, w, reference point)."""
    A = np.array(
        [[np.random.uniform(-1, 1) for _ in range(dimensions)] for _ in range(dimensions)]
    )
    A = A / np.linalg.norm(A)
    w = np.array([np.random.uniform(0, 1000) for _ in range(dimensions)])
    w = 100 * w / np.linalg.norm(w)
    point = [np.random.uniform(-k / 2, k / 2) for _ in range(dimensions)]
    return {"A": A, "w": w, "point": point}


def trial_source_from_pickle(pickle_path):
    """
    Replay trial parameters from Data_trials_d_*.p.

    The file is a list ordered like grid_sizes (highest resolution first); each
    block is a list of [A, w, point] per trial. Call set_resolution_index(i)
    before scoring each resolution.
    """
    all_trials = pickle.load(open(pickle_path, "rb"))
    resolution_index = [0]

    def source(dimensions, number_of_points, trial_number):
        record = all_trials[resolution_index[0]][trial_number]
        return {"A": record[0], "w": record[1], "point": record[2]}

    def set_resolution_index(idx):
        resolution_index[0] = idx

    source.set_resolution_index = set_resolution_index
    return source


def make_default_loss_function(k=1, trial_source=None, record_trials=True, d=None):
    """
    Build the manuscript's default loss: random quadratic f per trial.

    Parameters
    ----------
    k : float
        Hyperparameter cube side length (used when sampling trials).
    trial_source : callable(dimensions, number_of_points, trial_number) -> trial_state, optional
        Supplies trial_state for each (resolution, trial). By default, loads
        Data_trials_d_{d}.p from this script's directory if present; otherwise
        draws random A, w, point.
    record_trials : bool
        If True, attach .trial_cache for saving Data_trials_d_{d}.p.
    d : int, optional
        Dimensionality (needed to locate Data_trials_d_{d}.p).

    Returns
    -------
    loss_function
        Callable(point_pattern, dimensions, number_of_points, trial_number) -> float.
        When trials are loaded from pickle, also has set_resolution_index(i).
    """
    if trial_source is None:
        data_trials_path = BASE_DIR / "Data_trials_d_{}.p".format(d) if d is not None else None
        if data_trials_path is not None and data_trials_path.is_file():
            trial_source = trial_source_from_pickle(data_trials_path)
        else:

            def trial_source(dimensions, number_of_points, trial_number):
                return _sample_random_quadratic_trial(dimensions, k)

    trial_cache = {}

    def loss_function(point_pattern, dimensions, number_of_points, trial_number, k=1):
        key = (dimensions, number_of_points, trial_number)
        if key not in trial_cache:
            trial_cache[key] = trial_source(dimensions, number_of_points, trial_number)
        trial_state = trial_cache[key]
        point = trial_state["point"]
        A = trial_state["A"]
        w = trial_state["w"]
        dim = len(point)

        def objective(x):
            return np.linalg.norm(
                np.dot([x[i] - point[i] for i in range(dim)], A) * w
            ) ** 2

        best = objective(point_pattern[0])
        for x in point_pattern:
            val = objective(x)
            if val < best:
                best = val
        return best

    if record_trials:
        loss_function.trial_cache = trial_cache

    if hasattr(trial_source, "set_resolution_index"):
        _set_resolution_index = trial_source.set_resolution_index

        def set_resolution_index(idx):
            _set_resolution_index(idx)
            trial_cache.clear()

        loss_function.set_resolution_index = set_resolution_index

    return loss_function


def generate_pattern(ctx, make_fn, number=None, **kwargs):
    """
    Call a pattern generator with the shared signature:
    make_fn(number, k, d, trial_number=..., **kwargs).
    """
    return make_fn(
        ctx["target_points"] if number is None else number,
        ctx["k"],
        ctx["d"],
        trial_number=ctx.get("trial_number", 0),
        **kwargs,
    )


def _normalize_pattern_specs(patterns):
    """Accept (name, fn) or (name, fn, each_trial_bool|dict)."""
    normalized = []
    for item in patterns:
        if len(item) == 2:
            name, generator = item
            each_trial = False
        elif len(item) == 3:
            name, generator, opts = item
            if isinstance(opts, dict):
                each_trial = opts.get("each_trial", False)
            else:
                each_trial = bool(opts)
        else:
            raise ValueError(
                "Each pattern must be (name, generator) or (name, generator, each_trial)."
            )
        normalized.append({"name": name, "generator": generator, "each_trial": each_trial})
    return normalized


def run_pattern_comparison(
    patterns,
    d,
    k=1,
    trials=10000,
    grid_sizes=None,
    loss_function=None,
    output_dir=None,
    script_dir=None,
):
    """
    Compare point-pattern generators on a benchmark defined by loss_function.

    Parameters
    ----------
    patterns : list of (name, generator) or (name, generator, each_trial)
        generator(ctx) -> point pattern (list of coordinate vectors) or falsy if skipped.
        ctx keys: n, d, k, target_points, trial_number, cache.
        Pattern makers use generate_pattern(ctx, make_fn) or the same
        (number, k, d, trial_number=...) signature; trial_number selects
        hyperuniform files and is ignored by other generators.
    d : int
        Number of dimensions.
    k : float
        Side length of the hyperparameter cube (passed to default loss / pattern generators).
    trials : int
        Number of objective instances per grid resolution.
    grid_sizes : list of int, optional
        Values n such that target point count is n**d. Defaults to N-1 .. 1 with N=20 (d=2) or 5.
    loss_function : callable(point_pattern, dimensions, number_of_points, trial_number) -> float
        Score for one pattern on one trial. Default: quadratic benchmark via
        make_default_loss_function(k), loading Data_trials_d_{d}.p from this
        script's directory when present, otherwise generating new trials.
    output_dir : str, optional
        Defaults to run_YYYYMMDD_HHMMSS under script_dir.
    script_dir : str, optional
        Directory for the run folder; defaults to this file's directory.

    Returns
    -------
    dict
        Full results including per-resolution summaries and output paths.
    """
    pattern_specs = _normalize_pattern_specs(patterns)
    pattern_names = [spec["name"] for spec in pattern_specs]

    if grid_sizes is None:
        N = 20 if d == 2 else 5
        grid_sizes = [(N - i) for i in range(N - 1)]

    if loss_function is None:
        loss_function = make_default_loss_function(k=k, d=d)

    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(script_dir, "run_" + timestamp)
    os.makedirs(output_dir, exist_ok=True)

    wins_total = {name: 0 for name in pattern_names}
    everything_scores = []
    everything_numbers = {}
    everything_means = {}
    everything_stderr = {}
    data_trials = []
    by_grid_size = []

    for resolution_index, n in enumerate(grid_sizes):
        if hasattr(loss_function, "set_resolution_index"):
            loss_function.set_resolution_index(resolution_index)
        target_points = n ** d
        ctx_base = {
            "n": n,
            "d": d,
            "k": k,
            "target_points": target_points,
            "trial_number": 0,
            "cache": {},
        }

        static_tilings = {}
        for spec in pattern_specs:
            if spec["each_trial"]:
                continue
            tiling = spec["generator"](ctx_base)
            static_tilings[spec["name"]] = tiling if tiling else []

        scores = {name: [] for name in pattern_names}
        point_counts = {}
        data_trials_temp = []

        for trial_number in range(trials):
            ctx = {
                **ctx_base,
                "trial_number": trial_number,
            }

            trial_scores = {}
            for spec in pattern_specs:
                name = spec["name"]
                if spec["each_trial"]:
                    tiling = spec["generator"](ctx)
                else:
                    tiling = static_tilings.get(name, [])

                if tiling:
                    num_points = len(tiling)
                    point_counts[name] = num_points
                    score = loss_function(tiling, d, num_points, trial_number, k = k)
                    scores[name].append(score)
                    trial_scores[name] = score

            if trial_scores:
                winner = min(trial_scores, key=trial_scores.get)
                wins_total[winner] += 1

        resolution_summary = {"patterns": {}}
        scores_this_n = {}
        numbers_this_n = {}

        for name in pattern_names:
            if not scores[name]:
                continue
            mean_score = float(np.mean(scores[name]))
            stderr = float(np.std(scores[name]) / np.sqrt(len(scores[name])))
            num_points = point_counts.get(name, 0)

            scores_this_n[name] = scores[name]
            numbers_this_n[name] = num_points
            everything_means.setdefault(name, []).append(mean_score)
            everything_stderr.setdefault(name, []).append(stderr)
            everything_numbers.setdefault(name, []).append(num_points)

            resolution_summary["patterns"][name] = {
                "mean": mean_score,
                "stderr": stderr,
                "num_points": num_points,
                "n_trials_scored": len(scores[name]),
            }

        if hasattr(loss_function, "trial_cache"):
            ref_points = point_counts.get(pattern_names[0], target_points)
            for trial_number in range(trials):
                key = (d, ref_points, trial_number)
                if key in loss_function.trial_cache:
                    trial_state = loss_function.trial_cache[key]
                    # Legacy Data_trials_d_*.p format: [A, w, point]
                    data_trials_temp.append(
                        [trial_state["A"], trial_state["w"], trial_state["point"]]
                    )

        everything_scores.append(scores_this_n)
        data_trials.append(data_trials_temp)
        by_grid_size.append(
            {
                "n": n,
                "target_points": target_points,
                "reference_num_points": point_counts.get(pattern_names[0], target_points),
                **resolution_summary,
            }
        )

        ref_pts = resolution_summary.get("reference_num_points", target_points)
        print("Comparison for", ref_pts, "points finished")

    results = {
        "metadata": {
            "d": d,
            "k": k,
            "trials": trials,
            "grid_sizes": list(grid_sizes),
            "pattern_names": pattern_names,
            "wins": wins_total,
            "timestamp": datetime.now().isoformat(),
        },
        "by_grid_size": by_grid_size,
        "everything_scores": everything_scores,
        "everything_numbers": everything_numbers,
        "everything_means": everything_means,
        "everything_stderr": everything_stderr,
        "data_trials": data_trials,
    }

    prefix = os.path.join(output_dir, "")
    pickle.dump(data_trials, open(prefix + "Data_trials_d_" + str(d) + ".p", "wb"))
    pickle.dump(everything_scores, open(prefix + "Scores_d_" + str(d) + ".pkl", "wb"))
    pickle.dump(everything_numbers, open(prefix + "Numbers_d_" + str(d) + ".pkl", "wb"))
    pickle.dump(results, open(prefix + "results_d_" + str(d) + ".pkl", "wb"))

    results["output_dir"] = output_dir
    results["output_files"] = {
        "data_trials": prefix + "Data_trials_d_" + str(d) + ".p",
        "scores": prefix + "Scores_d_" + str(d) + ".pkl",
        "numbers": prefix + "Numbers_d_" + str(d) + ".pkl",
        "full_results": prefix + "results_d_" + str(d) + ".pkl",
    }
    print("Comparison finished. Results saved to", output_dir)
    return results


def build_default_patterns(d, k, mak, incl, stealth=0.49):
    old_min = np.array([0 for _ in range(d)])
    old_max = np.array([k for _ in range(d)])
    new_min = np.array([-k / 2 for _ in range(d)])
    new_max = np.array([k / 2 for _ in range(d)])

    patterns = [
        ("grid", lambda ctx: generate_pattern(ctx, make_grid)),
        ("random", lambda ctx: generate_pattern(ctx, make_random, number=len(ctx["cache"]["rsa"])), True),
        ("gridrandom", lambda ctx: generate_pattern(ctx, make_gridrandom), True),
    ]

    for star_count in range(4, mak + 1):
        name = "quasi_" + str(star_count) + "star"
        def quasi_gen(ctx, t=star_count):
            if name not in ctx["cache"]:
                vectors = make_vectors(ctx["d"], t)
                ctx["cache"][name] = make_quasi_nD(
                    ctx["target_points"], ctx["k"], ctx["d"], t, vectors
                )
            return ctx["cache"][name]
        patterns.append((name, quasi_gen))

    def rsa_gen(ctx):
        if "rsa" not in ctx["cache"]:
            rsa = make_RSA(ctx["target_points"], ctx["k"], ctx["d"])
            transform_tiling(rsa, old_min, old_max, new_min, new_max, ctx["d"])
            ctx["cache"]["rsa"] = rsa
        return ctx["cache"]["rsa"]

    patterns.append(("RSA", rsa_gen))
    patterns.append(
        ("latin", lambda ctx: generate_pattern(ctx, make_latin), True)
    )

    if incl:
        patterns.append(
            ("hyperuniform", lambda ctx: generate_pattern(ctx, make_hyperuniform, stealth=stealth), True)
        )

    return patterns


def rastrigin_reduction_formula(dim):
    """How many dimensions are actually important for the Rastrigin function?"""
    if dim == 2:
        return 1
    return dim - 2


def grid_sizes_for_dimension(d):
    N = 20 if d == 2 else 5
    return [(N - i) for i in range(N - 1)]


def build_comparison_patterns(d):
    patterns_old = [
        ("grid", lambda ctx: generate_pattern(ctx, make_grid)),
        ("random", lambda ctx: generate_pattern(ctx, make_random), True),
        ("gridrandom", lambda ctx: generate_pattern(ctx, make_gridrandom), True),
        ("latin", lambda ctx: generate_pattern(ctx, make_latin), True),
    ]

    patterns_sobol_halton = [
        ("sobol", lambda ctx: generate_pattern(ctx, make_sobol), True),
        ("halton", lambda ctx: generate_pattern(ctx, make_halton), True),
    ]

    patterns_hyper = []
    if d == 2:
        # stealth_values = [0.49, 0.4]
        stealth_values = [0.4, 0.3]
    else:
        stealth_values = [0.4, 0.3]

    for stealth in stealth_values:
        patterns_hyper.append(
            (
                f"hyperuniform_{stealth}",
                lambda ctx, s=stealth: generate_pattern(ctx, make_hyperuniform, stealth=s),
                True,
            )
        )

    return patterns_hyper, patterns_old, patterns_sobol_halton


BENCHMARK_ALIASES = {
    "syblinski": "styblinski_tang",
    "syblinski_tang": "styblinski_tang",
    "styblinski": "styblinski_tang",
}


BENCHMARKS = {
    "simple": {
        "title": "simple generic function",
        "loss": loss_generic,
    },
    "rastrigin": {
        "title": "Rastrigin function",
        "loss": loss_rastrigin,
    },
    "branin": {
        "title": "reduced Branin function",
        "loss": loss_branin,
    },
    "langermann": {
        "title": "Langermann function",
        "loss": loss_langermann,
    },
    "styblinski_tang": {
        "title": "Styblinski-Tang function",
        "loss": loss_styblinski_tang,
    },
    "zakharov": {
        "title": "Zakharov function",
        "loss": loss_zakharov,
    },
    "schwefel": {
        "title": "Schwefel function",
        "loss": loss_schwefel,
    },
}


def normalize_benchmark_name(name):
    key = name.lower().replace("-", "_").replace(" ", "_")
    return BENCHMARK_ALIASES.get(key, key)


def resolve_benchmarks(function_names):
    resolved = []
    for name in function_names:
        key = normalize_benchmark_name(name)
        if key not in BENCHMARKS:
            valid = sorted(BENCHMARKS)
            raise ValueError(f"Unknown function {name!r}. Choose from: {', '.join(valid)}")
        if key not in resolved:
            resolved.append(key)
    return resolved


def random_rotation_matrix(d, seed):
    """Haar-random element of SO(d)."""
    rng = np.random.RandomState(seed)
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def apply_rotation(loss_fn, rotate=False):
    """Optionally rotate points by a random SO(d) matrix, then score as loss_fn."""
    if not rotate:
        return loss_fn

    def rotated_loss(point_pattern, dimensions, number_of_points, trial_number, k=1, **kwargs):
        R = random_rotation_matrix(dimensions, trial_number)
        rotated = (np.asarray(point_pattern, dtype=float) @ R.T).tolist()
        return loss_fn(rotated, dimensions, number_of_points, trial_number, k=k, **kwargs)

    for attr in ("trial_cache", "set_resolution_index"):
        if hasattr(loss_fn, attr):
            setattr(rotated_loss, attr, getattr(loss_fn, attr))
    return rotated_loss


def loss_function_for_benchmark(benchmark_id, d, k=1, rotate=False):
    return apply_rotation(BENCHMARKS[benchmark_id]["loss"], rotate=rotate)


def patterns_for_benchmark(benchmark_id, patterns_hyper, patterns_old, patterns_sobol_halton):
    if BENCHMARKS[benchmark_id].get("hyper_only"):
        return patterns_hyper
    return patterns_hyper + patterns_old + patterns_sobol_halton


def run_selected_benchmarks(dimensions, function_names, run="run_09_04", k=1, trials=10000, rotation=False):
    benchmark_ids = resolve_benchmarks(function_names)

    for d in dimensions:
        patterns_hyper, patterns_old, patterns_sobol_halton = build_comparison_patterns(d)
        grid_sizes = grid_sizes_for_dimension(d)

        for benchmark_id in benchmark_ids:
            spec = BENCHMARKS[benchmark_id]
            allowed_dims = spec.get("dimensions")
            if allowed_dims is not None and d not in allowed_dims:
                print(
                    f"Skipping {benchmark_id} for {d}D "
                    f"(only supported for dimensions {list(allowed_dims)})"
                )
                continue

            out_name = f"{benchmark_id}_rotated" if rotation else benchmark_id
            print(
                f"\n\n-----------Running the comparison for the "
                f"{spec['title']} ({d}D){', rotated' if rotation else ''}-----------"
            )
            run_pattern_comparison(
                patterns=patterns_for_benchmark(
                    benchmark_id, patterns_hyper, patterns_old, patterns_sobol_halton
                ),
                d=d,
                k=k,
                trials=trials,
                grid_sizes=grid_sizes,
                loss_function=loss_function_for_benchmark(benchmark_id, d, k=k, rotate=rotation),
                output_dir=f"{run}/{out_name}",
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run point-pattern comparisons for selected benchmark functions "
            "and dimensions."
        )
    )
    parser.add_argument(
        "--dimensions",
        "-d",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
        help="Dimension numbers to run (default: 2 3 4 5).",
    )
    parser.add_argument(
        "--functions",
        "-f",
        nargs="+",
        default=[
            "simple",
            "rastrigin",
            "branin",
            "langermann",
            "styblinski_tang",
            "zakharov",
            "schwefel",
        ],
        help=(
            "Benchmark functions to run. Choices: "
            + ", ".join(sorted(BENCHMARKS))
            + ". Aliases: syblinski, styblinski."
        ),
    )
    parser.add_argument(
        "--run",
        default="results",
        help="Output run directory prefix (results go to RUN/FUNC-ID).",
    )
    parser.add_argument(
        "-rotation",
        "--rotation",
        action="store_true",
        help="Apply a random rotation to evaluation points before scoring.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10000,
        help="Number of objective instances per grid resolution.",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=1,
        help="Side length of the hyperparameter cube.",
    )
    return parser.parse_args()


################making the comparison


if __name__ == "__main__":
    args = parse_args()
    run_selected_benchmarks(
        dimensions=args.dimensions,
        function_names=args.functions,
        run=args.run,
        k=args.k,
        trials=args.trials,
        rotation=args.rotation,
    )