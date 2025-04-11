import numpy as np
import matplotlib.pyplot as plt
from numba import njit
#==============================================================================
# Global Parametrs
#==============================================================================
NUM_AVALANCHES_TOTAL = int(1e6)  # How many avalanches to record per lattice size
EQUILIBRATE          = int(1e4)  # How many grain additions for burn-in
SYSTEM_SIZES         = [128,256]       # The lattice sizes we'll simulate
# Paper exponents for final data collapse
BETA_S, TAU_S = 2.70, 1.27
BETA_A, TAU_A = 2.00, 1.35
BETA_T, TAU_T = 1.50, 1.50

# Number of histogram bins for log-binned distributions
NBINS = 30

# q range for moment-spectra analysis
Q_VALUES = np.linspace(0.0, 2.0, 21)

# Random number array size (we periodically refill it when we run out)
RNG_CHUNK_SIZE = int(2e6)

#==============================================================================
# 1) MODIFIED STABILIZE FUNCTION (Synchronous Update, Numba-compiled)
#==============================================================================
@njit
def stabilize(heights, L, rng_array, rng_index):
    total_topplings = 0
    duration = 0
    toppled_flag = np.zeros((L, L), dtype=np.bool_)
    directions = ((-1,0), (1,0), (0,-1), (0,1))
    
    # Preallocate arrays to store indices of unstable sites (max L*L sites)
    unstable_i = np.empty(L*L, dtype=np.int64)
    unstable_j = np.empty(L*L, dtype=np.int64)
    
    while True:
        # Identify all sites unstable at the beginning of the pass.
        count = 0
        for i in range(L):
            for j in range(L):
                if heights[i, j] >= 2:
                    unstable_i[count] = i
                    unstable_j[count] = j
                    count += 1
        if count == 0:
            break  # No unstable sites remain.
        
        duration += 1
        
        # Topple each unstable site exactly once.
        for idx in range(count):
            i = unstable_i[idx]
            j = unstable_j[idx]
            toppled_flag[i, j] = True
            heights[i, j] -= 2
            total_topplings += 1
        
        # Distribute the grains from these topplings (after all topplings in the pass).
        for idx in range(count):
            i = unstable_i[idx]
            j = unstable_j[idx]
            for _ in range(2):
                if rng_index >= rng_array.size:
                    rng_array[:] = np.random.rand(rng_array.size)
                    rng_index = 0
                r = rng_array[rng_index]
                rng_index += 1
                d_idx = int(r * 4)
                dx, dy = directions[d_idx]
                ni = i + dx
                nj = j + dy
                if 0 <= ni < L and 0 <= nj < L:
                    heights[ni, nj] += 1
    avalanche_area = np.count_nonzero(toppled_flag)
    return total_topplings, avalanche_area, duration, rng_index

#==============================================================================
# 2) SIMULATE FUNCTION (Numba-compiled)
#==============================================================================
@njit
def simulate_sandpile(L, num_avalanches, equilibrate, rng_array):
    """
    Simulate the 2D stochastic sandpile model on an LxL lattice:
      1) Initialize each site with 1 grain.
      2) Equilibrate by adding grains randomly (not recorded).
      3) Record s, a, t for 'num_avalanches' subsequent grain additions.
    """
    heights = np.ones((L, L), dtype=np.int64)
    rng_index = 0
    
    # Equilibration phase
    for _ in range(equilibrate):
        if rng_index + 2 >= rng_array.size:
            rng_array[:] = np.random.rand(rng_array.size)
            rng_index = 0
        i_rand = int(rng_array[rng_index] * L)
        j_rand = int(rng_array[rng_index+1] * L)
        rng_index += 2
        
        heights[i_rand, j_rand] += 1
        _, _, _, rng_index = stabilize(heights, L, rng_array, rng_index)
    
    # Measurement phase
    s_vals = np.empty(num_avalanches, dtype=np.float64)
    a_vals = np.empty(num_avalanches, dtype=np.float64)
    t_vals = np.empty(num_avalanches, dtype=np.float64)
    
    for idx in range(num_avalanches):
        if rng_index + 2 >= rng_array.size:
            rng_array[:] = np.random.rand(rng_array.size)
            rng_index = 0
        i_rand = int(rng_array[rng_index] * L)
        j_rand = int(rng_array[rng_index+1] * L)
        rng_index += 2
        heights[i_rand, j_rand] += 1
        
        s, a, t, rng_index = stabilize(heights, L, rng_array, rng_index)
        s_vals[idx] = s
        a_vals[idx] = a
        t_vals[idx] = t
    
    return s_vals, a_vals, t_vals

#==============================================================================
# 3) RUN MULTIPLE L
#==============================================================================
def run_sandpile_for_sizes(sizes, n_avals, eq_steps):
    """
    Run the 2D stochastic sandpile for each L in 'sizes' and return a dict with s,a,t data.
    """
    data = {}
    rng_array = np.random.rand(RNG_CHUNK_SIZE).astype(np.float64)
    for L in sizes:
        print(f"Running L={L} ...")
        s_vals, a_vals, t_vals = simulate_sandpile(L, n_avals, eq_steps, rng_array)
        data[L] = dict(s=s_vals, a=a_vals, t=t_vals)
    return data

#==============================================================================
# 4) L MOMENT SPECTRUM
#==============================================================================
def compute_L_moment_spectrum(x, L, q_values):
    """
    For a single lattice size L and avalanche data x,
    define the L moment spectrum:
      sigma_L(q) = ln( <x^q> ) / ln(L).
    """
    result = []
    for q in q_values:
        mean_val = np.mean(x**q)
        if mean_val <= 0.0:
            result.append(np.nan)
        else:
            result.append(np.log(mean_val) / np.log(L))
    return np.array(result)

def plot_L_moment_spectra(data_dict, obs='s', q_values=Q_VALUES):
    """
    Plot the L moment spectrum for each lattice size.
    One line per size L in data_dict.
    Also displays the total number of avalanches (N).
    """
    fig, ax = plt.subplots(figsize=(7,5))
    all_L = sorted(data_dict.keys())
    markers = ['o','s','d','v','^','<','>']
    for i, L in enumerate(all_L):
        x = data_dict[L][obs]
        sig = compute_L_moment_spectrum(x, L, q_values)
        m = markers[i % len(markers)]
        ax.plot(q_values, sig, m+'-', label=f"L={L}")
    ax.set_xlabel("q")
    ax.set_ylabel(r"$\sigma(q)$")
    ax.set_title(f"L Moment Spectrum for '{obs}'")
    ax.text(0.05, 0.95, f"N = {NUM_AVALANCHES_TOTAL}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', color='black')
    ax.legend()
    plt.tight_layout()
    plt.show()

#==============================================================================
# 5) LOG-BINNED DATA COLLAPSE
#==============================================================================
def log_binned_hist(x, nbins=30):
    """
    Build a log-spaced histogram of x>0 and return (bin_centers, prob_density).
    p = hist/(count * bin_width).
    """
    x = x[x>0]
    if len(x) == 0:
        return None, None
    x_min, x_max = x.min(), x.max()
    if x_min <= 0:
        x_min = 1e-9
    bins = np.logspace(np.log10(x_min), np.log10(x_max), nbins+1)
    hist, edges = np.histogram(x, bins=bins)
    bin_centers = np.sqrt(edges[:-1]*edges[1:])
    widths = edges[1:] - edges[:-1]
    total_count = len(x)
    p = hist/(total_count*widths)
    return bin_centers, p

def data_collapse_plot(data_dict, obs, beta, tau, label):
    """
    Data collapse plot:
      x_scaled = x / L^beta
      p_scaled = p(x)*L^(beta*tau)
    Creates a log-binned histogram, rescales and plots.
    """
    fig, ax = plt.subplots(figsize=(6,5))
    for L in sorted(data_dict.keys()):
        x = data_dict[L][obs]
        bc, p = log_binned_hist(x, nbins=NBINS)
        if bc is None:
            continue
        x_scaled = bc / (L**beta)
        p_scaled = p * (L**(beta*tau))
        ax.plot(x_scaled, p_scaled, 'o', ms=4, label=f"L={L}")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$x / L^{\beta}$")
    ax.set_ylabel(r"$P(x)\, L^{\beta \tau}$")
    ax.set_title(f"Data Collapse: {label}  (β={beta:.2f}, τ={tau:.2f})")
    ax.text(0.05, 0.95, f"N = {NUM_AVALANCHES_TOTAL}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', color='black')
    ax.legend()
    plt.tight_layout()
    plt.show()

#==============================================================================
# MAIN
#==============================================================================
def main():
    data = run_sandpile_for_sizes(SYSTEM_SIZES, NUM_AVALANCHES_TOTAL, EQUILIBRATE)
    for obs_label in ['s','a','t']:
        plot_L_moment_spectra(data, obs=obs_label, q_values=Q_VALUES)
    data_collapse_plot(data, 's', BETA_S, TAU_S, label='Size s')
    data_collapse_plot(data, 'a', BETA_A, TAU_A, label='Area a')
    data_collapse_plot(data, 't', BETA_T, TAU_T, label='Duration t')
    print("\nBasic Stats:")
    for L in SYSTEM_SIZES:
        s_mean = np.mean(data[L]['s'])
        a_mean = np.mean(data[L]['a'])
        t_mean = np.mean(data[L]['t'])
        print(f"L={L}: mean(s)={s_mean:.2f}, mean(a)={a_mean:.2f}, mean(t)={t_mean:.2f}")

if __name__ == "__main__":
    main()
