"""
Self-contained script for:
  1) Simulating the 2D stochastic sandpile model at multiple lattice sizes
  2) Computing avalanche observables (s, a, t)
  3) Generating L moment-spectra plots for each lattice size
  4) Doing data-collapse plots for the distributions
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

#==============================================================================
# Global Parameters
#==============================================================================
NUM_AVALANCHES_TOTAL = int(1e6)  # How many avalanches to record per lattice size
EQUILIBRATE          = int(1e4)  # How many grain additions for burn-in
SYSTEM_SIZES         = [128, 256]       # The lattice sizes we'll simulate
LAMBDA_RATE          = 0.5

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
# 1) STABILIZE FUNCTION (Numba-compiled)
#==============================================================================
@njit
def stabilize(arr, lenArr, addIndex, rate):
    """
    Repeatedly topple sites in 'heights' until all are < 2.
    Each toppling subtracts 2 grains from a site, distributing
    them randomly among neighbors (with boundary dissipation).

    Returns:
      total_topplings (s),
      avalanche_area  (a),
      duration       (t),
      updated_rng_index
    """
    topple = True
    numTopples = 0
    duration = 0
    toppled_flag = np.zeros(len(arr))
    
    ## generate batch of instructions for speed
    #instr = np.random.choice(np.arange(-1,2), size = 100*lenArr, p = [(1)/(2*(1+rate)), (rate)/(1+rate), (1)/(2*(1+rate))]).tolist()
    instr = np.random.choice(np.arange(-1, 2), size = 100*lenArr)

    ## add 1 particle to de-stabilize, and set index to active
    arr[addIndex][1] += 1 ## add one to the number of particles
    arr[addIndex][0] = 1 ## change state to be awake

    while topple:
        roundTopples = 0
        duration += 1
        for i in range(lenArr):
            currIndex = arr[i]
            if currIndex[0] == 1 and currIndex[1] > 0:
                toppled_flag[i] = 1
                if len(instr) < 1: ## if instruction list was empty, replenish, then grab instruction for index
                    #instr = np.random.choice(np.arange(-1,2), size = 100*lenArr, p = [(1)/(2 +rate), (rate)/(1+rate), (1)/(2+rate)]).tolist()
                    instr = np.random.choice(np.arange(-1, 2), size = 100*lenArr)

                #currInstr = instr.pop() ## generate instruction for current index

                currInstr = instr[0]
                instr = instr[1:]

                roundTopples += 1
                numTopples += 1

                if currInstr == 0 and currIndex[1] == 1:
                    currIndex[0] = 0
                elif currInstr == -1 or currInstr == 1: ## left or right
                    currIndex[1] -= 1
                    currIndex[0] = 0 if currIndex[1] < 1 else 1

                    ## check to make sure particle is added to index in range of arr
                    if (i + currInstr) < (lenArr) and (i + currInstr) >= 0:
                        arr[i + currInstr][1] += 1
                        arr[i + currInstr][0] = 1
                        

            topple = True if roundTopples > 0 else False ## topple is false if no indices needed to be stabilized

    
    ## now stabilization is complete - exits after one round of no need for stabilizing
    ## ie. all indices were stable in the pass through
    avalanche_area = np.count_nonzero(toppled_flag)
    return numTopples, avalanche_area, duration



#==============================================================================
# 2) SIMULATE FUNCTION (Numba-compiled)
#==============================================================================
@njit
def simulate_sandpile(L, num_avalanches, equilibrate, lambda_rate):
    """
    Simulate the 2D stochastic sandpile model on an LxL lattice:
      1) Initialize each site with 1 grain.
      2) Equilibrate by adding grains randomly (not recorded).
      3) Record s, a, t for 'num_avalanches' subsequent grain additions.
    """
    # All sites start with 1 grain
    arr = []
    for i in range(L):
        particle = [] ## first value in list is state, second is number of particles
        particle.append(1) ## awake = 1, sleep = 0
        particle.append(1)
        arr.append(particle)

    # Equilibration phase
    for _ in range(equilibrate):
        _, _, _ = stabilize(arr, len(arr), int(len(arr) / 2), lambda_rate)

    ## measurement phase
    s_vals = np.empty(num_avalanches, dtype=np.float64)
    a_vals = np.empty(num_avalanches, dtype=np.float64)
    t_vals = np.empty(num_avalanches, dtype=np.float64)

    for i in range(num_avalanches):
        s, a, t = stabilize(arr, len(arr), int(len(arr) / 2), lambda_rate)
        s_vals[i] = s
        a_vals[i] = a
        t_vals[i] = t
    
    return s_vals, a_vals, t_vals

#==============================================================================
# 3) RUN MULTIPLE L
#==============================================================================
def run_sandpile_for_sizes(sizes, n_avals, eq_steps, lambda_rate):
    """
    Run the 2D stochastic sandpile for each L in 'sizes' and return a dict with s,a,t data.
    """
    data = {}
    for L in sizes:
        print(f"Running L={L} ...")
        s_vals, a_vals, t_vals = simulate_sandpile(L, n_avals, eq_steps, lambda_rate)
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
            # sigma_L(q) = ln(<x^q>) / ln(L)
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
    markers = ['o','s','d','v','^','<','>']  # for variety if you have many sizes
    
    for i, L in enumerate(all_L):
        x = data_dict[L][obs]
        # Compute sigma_L(q) for this L
        sig = compute_L_moment_spectrum(x, L, q_values)
        m = markers[i % len(markers)]
        ax.plot(q_values, sig, m+'-', label=f"L={L}")
    
    ax.set_xlabel("q")
    ax.set_ylabel(r"$\sigma(q)$")
    ax.set_title(f"L Moment Spectrum for '{obs}'")
    # Annotate with avalanche count
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

    Creates a log-binned histogram, then rescales and plots.
    Also displays the total number of avalanches (N).
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
    # Annotate with avalanche count
    ax.text(0.05, 0.95, f"N = {NUM_AVALANCHES_TOTAL}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', color='black')
    ax.legend()
    plt.tight_layout()
    plt.show()

#==============================================================================
# MAIN
#==============================================================================
def main():
    # 1) Run simulation for each L
    data = run_sandpile_for_sizes(SYSTEM_SIZES, NUM_AVALANCHES_TOTAL, EQUILIBRATE, LAMBDA_RATE)
    
    # 2) For each observable (s, a, t), plot the L moment spectrum
    for obs_label in ['s','a','t']:
        plot_L_moment_spectra(data, obs=obs_label, q_values=Q_VALUES)
    
    # 3) Example data collapse using known/paper exponents
    data_collapse_plot(data, 's', BETA_S, TAU_S, label='Size s')
    data_collapse_plot(data, 'a', BETA_A, TAU_A, label='Area a')
    data_collapse_plot(data, 't', BETA_T, TAU_T, label='Duration t')
    
    # 4) Basic stats
    print("\nBasic Stats:")
    for L in SYSTEM_SIZES:
        s_mean = np.mean(data[L]['s'])
        a_mean = np.mean(data[L]['a'])
        t_mean = np.mean(data[L]['t'])
        print(f"L={L}: mean(s)={s_mean:.2f}, mean(a)={a_mean:.2f}, mean(t)={t_mean:.2f}")

if __name__ == "__main__":
    main()
