!pip install numba
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


## creates two arrays of length n
## One to represent a particle at each site
## One to represent that that site is awake (set to True)
@njit
def setUp(n: int):
    part_arr = np.ones(n, dtype=np.int16)
    state_arr = np.full(n, True, dtype=np.bool_)
    return part_arr, state_arr

## Goes through array and moves particles until stabilization (one asleep particle or none per site)
@njit
def runRound(part_arr, state_arr, addIndex, lenArr, directions): 
    topple = True
    numTopples = 0
    time = 0
    toppled_flag = np.zeros(lenArr, dtype=np.bool_)

    ## generate batch of instructions for speed
    instr = np.random.rand(100*lenArr)
    instr_in = 0

    ## add 1 particle to middle to de-stabilize, and set index to active
    part_arr[addIndex] += 1
    state_arr[addIndex] = True

    while topple:
        roundTopples = 0
        time += 1
        for i in range(lenArr):
            while state_arr[i] == True and part_arr[i] > 0:
               ## if instruction list was empty, replenish, then grab instruction for index
                if instr_in >= instr.size:
                    instr[:] = np.random.rand(100*lenArr)
                    instr_in = 0

                currInstr = directions[int(instr[instr_in]*len(directions))] ## get instruction to move particle
                instr_in += 1
                toppled_flag[i] = True ## Record if a site was toppled
                numTopples += 1
                roundTopples += 1

                if currInstr == 0 and part_arr[i] == 1: ## Site falls asleep
                    state_arr[i] = False 
                elif currInstr == -1 or currInstr == 1: ## particle moves left or right
                    part_arr[i] -= 1
                    state_arr[i] = False if part_arr[i] < 1 else True
                    ## check to make sure particle is added to index in range of arr
                    if (i + currInstr) < (lenArr) and (i + currInstr) >= 0:
                        part_arr[i + currInstr] += 1
                        state_arr[i + currInstr] = True

            topple = True if roundTopples > 0 else False ## topple is false if no indices needed to be stabilized

    ## now stabilization is complete - exits after one round of no need for stabilizing
    ## ie. all indices were stable in the pass through
    #numPar = sum(part_arr)
    #density = numPar/lenArr    #Optional calculation of density (not used for calculation of critical exponent)
    
    return numTopples, np.count_nonzero(toppled_flag), time
   

## Runs a specified number of stabilization rounds at the requested particle amount and rate of lambda
def runNTrials(nParticles = 128, nTrials = 1000, lambda_rate = 0.5):
    part_arr, state_arr = setUp(nParticles)
    
    #avgDens = np.empty(nTrials, dtype=np.float64) ## optional density variable
    numTopples = np.empty(nTrials, dtype=np.float64)
    avalanche_area = np.empty(nTrials, dtype=np.float64)
    duration = np.empty(nTrials, dtype=np.float64)
    #Workaround to get correct distribution of instructions since numpy.random.choice doesn't work well:
    size_directions = int(1/(1/(2*(1+lambda_rate))))
    directions = np.zeros(size_directions, dtype=np.int16)
    directions[0] = -1
    directions[1] = 1

    for i in range(nTrials):
        #density, 
        topples, size, time = runRound(part_arr, state_arr, int(len(part_arr) / 2), len(part_arr), directions)
        #avgDens[i] = density
        numTopples[i] = topples 
        avalanche_area[i] = size
        duration[i] = time
        
    #return avgDens, 
    return numTopples, avalanche_area, duration

## All code below shamelessly copied from Hyojeong with small changes because of need to calculate beta and tau and make plots
## for two different lambdas    

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

def plot_L_moment_spectra(data_dict, obs, q_values, num_avalanches_total, rate):
    """
    Plot the L moment spectrum for each lattice size.
    One line per size L in data_dict.
    Also displays the total number of avalanches (N).
    """
    fig, ax = plt.subplots(figsize=(7,5))
    beta = {}
    tau = {}
    
    all_L = sorted(data_dict.keys())
    markers = ['o','s','d','v','^','<','>']  # for variety if you have many sizes
    
    for i, L in enumerate(all_L):
        x = data_dict[L][obs]
        # Compute sigma_L(q) for this L
        sig = compute_L_moment_spectrum(x, L, q_values)
        m = markers[i % len(markers)]
        b = int(((sig[-1] - sig[-4])/0.3)*1000)/1000 # calculate beta by taking the slope between the last and fourth to last moment
        beta[L] = b                                  # trimmed to 3 decimal places
        tau[L] = int((-1 * ((sig[10]/b) -2))*1000)/1000 # calculate tau using (2 − τx)βx = σx(1) with beta and the 
                                                        # first moment (in this case found at sig[10]) 
                                                        # trimmed to 3 decimal places
        ax.plot(q_values, sig, m+'-', label=f"L={L}")
    
    ax.set_xlabel("q")
    ax.set_ylabel(r"$\sigma(q)$")
    ax.set_title(f"L Moment Spectrum for '{obs}'")
    # Annotate with avalanche count, lamda, beta, and tau
    ax.text(0.05, 0.95, f"N = {num_avalanches_total}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', color='black')
    ax.text(0.05, 0.90, f"Lambda = {rate}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', color='black')
    ax.text(0.05, 0.85, f"Beta = {beta}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', color='black')
    ax.text(0.05, 0.80, f"Tau = {tau}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', color='black')
    ax.legend()
    plt.tight_layout()
    plt.show()
    return beta, tau

def log_binned_hist(x, nbins=60):
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

def data_collapse_plot(data_dict, obs, beta, tau, label, num_avalanches_total):
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
        b = beta[L]
        t = tau[L]
        bc, p = log_binned_hist(x, nbins=30)
        if bc is None:
            continue
        x_scaled = bc / (L**b)
        p_scaled = p * (L**(b*t))
        ax.plot(x_scaled, p_scaled, 'o', ms=4, label=f"L={L}")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$x / L^{\beta}$")
    ax.set_ylabel(r"$P(x)\, L^{\beta \tau}$")
    ax.set_title(f"Data Collapse: {label}  (β:{beta}, τ:{tau})")
    # Annotate with avalanche count and lambda
    ax.text(0.05, 0.95, f"N = {num_avalanches_total}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', color='black')
    ax.text(0.05, 0.90, f"lambda = {rate}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', color='black')
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
  sizes = [128, 256]
  trials = 10000
  rates = [0.5, 2]
  data = {}
  ## Produces 12 total plots. A moment spectrum and data collapse plot for s,a,t at 2 different rates of lambda
  for rate in rates:
    for size in sizes:
      topples, aval_size, duration = runNTrials(size, trials, rate)
      data[size] = dict(s=topples, a=aval_size, t=duration)
    for obs_label, title in zip(['s','a','t'], ["Size s", "Area a", "Duration t"]):
      beta, tau = plot_L_moment_spectra(data, obs_label, np.linspace(0.0, 2.0, 21), trials, rate)
      data_collapse_plot(data, obs_label, beta, tau, title, trials, rate)
    
if __name__ == "__main__":
    main()
