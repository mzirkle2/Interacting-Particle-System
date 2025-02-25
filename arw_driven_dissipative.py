import numpy as np
import matplotlib.pyplot as plt

## defining object to hold data for each index
## welcome to change - just ease of access/understanding rn
## default index object is awake with one particle
class index:
    def __init__(self, state = "A", numPar = 1):
        self.state = state
        self.numPar = numPar

    def __str__(self):
        return f"State: {self.state}, numPar = {self.numPar}"


## creates an array of length n
## with one particle at each index (already awake)
def setUp(n: int):
    arr = []
    for i in range(n):
        arr.append(index())

    return arr

def runRound(arr, addIndex, lenArr, rate):
    topple = True
    numTopples = 0
    
    ## generate batch of instructions for speed
    instr = np.random.choice(np.arange(-1,2), size = 100*lenArr, p = [(1)/(2*(1+rate)), (rate)/(1+rate), (1)/(2*(1+rate))]).tolist()

    ## add 1 particle to de-stabilize, and set index to active
    arr[addIndex].numPar += 1
    arr[addIndex].state = "A"

    while topple:
        roundTopples = 0
        for i in range(lenArr):
            currIndex = arr[i]
            if currIndex.state == "A" and currIndex.numPar > 0:
                if len(instr) < 1: ## if instruction list was empty, replenish, then grab instruction for index
                    instr = np.random.choice(np.arange(-1,2), size = 100*lenArr, p = [(1)/(2*(1+rate)), (rate)/(1+rate), (1)/(2*(1+rate))]).tolist()
                
                currInstr = instr.pop() ## generate instruction for current index


                if currInstr == 0 and currIndex.numPar == 1:
                    currIndex.state = "S"
                    numTopples += 1
                    roundTopples += 1
                elif currInstr == -1 or currInstr == 1: ## left or right
                    currIndex.numPar -= 1
                    currIndex.state = "S" if currIndex.numPar < 1 else "A"

                    ## check to make sure particle is added to index in range of arr
                    if (i + currInstr) < (lenArr) and (i + currInstr) >= 0:
                        arr[i + currInstr].numPar += 1
                        arr[i + currInstr].state = "A"
                        
                    numTopples += 1
                    roundTopples += 1

            topple = True if roundTopples > 0 else False ## topple is false if no indices needed to be stabilized

    
    ## now stabilization is complete - exits after one round of no need for stabilizing
    ## ie. all indices were stable in the pass through
    numPar = sum([obj.numPar for obj in arr])
    print(f"Average density: {numPar / lenArr}")
    return (numPar / lenArr), numTopples


## will run a specified number of rounds, where each round sees 1 active particle
## added, and will stabilize until all are asleep before repeating with another active particle
## and so on
## params
    ## nParticles - the length of starting array, where the very first round sees every index filled with exactly
                    ## one active particle
    ## nTrials - the number of times a single active particle is added after stabilization
    ## lambda_rate - the rate at which particles fall asleep
## returns
    ## avgDens - a list of densities at the end of each round of stabilization
    ## numTopples - a list of the number of instructions used in a round of stabilization
def runNTrials(nParticles = 100, nTrials = 100, lambda_rate = 0.5):
    arr = setUp(nParticles)

    avgDens = []
    numTopples = []

    for i in range(nTrials):
        density, topples = runRound(arr, int(len(arr) / 2), len(arr), lambda_rate)
        avgDens.append(density)
        numTopples.append(topples)


    return avgDens, numTopples


def sampleVar(nList):
    listParticles = []
    for n in nList:
        nVar = []
        for j in range(100):
            avgDens, _ = runNTrials(n, 1000)
            nVar.append([i * n for i in avgDens]) ## one entry in list is list of number of particles at end of each stabilization round
    
        listParticles.append(nVar)


    listParticles = [[np.var(items) for items in nNum] for nNum in listParticles]
    listParticles = [np.mean(x) for x in listParticles]
    return listParticles
    

if __name__ == "__main__":
    ntrials = 100
    nparticles = 100
    lambda_rate = 0.5
    #nList = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    nList = [1, 2, 4, 8, 16, 32, 64, 128, 150, 200, 256, 300]
    
    #avgDens, numTopples = runNTrials(nparticles, ntrials, lambda_rate)
    listvar = sampleVar(nList)

    print(listvar)

    plt.figure(1)
    plt.plot(np.array(nList), np.array(listvar))
    plt.xlabel("Number of Starting Particles")
    plt.ylabel(f"Variance of Number of Particles")
    plt.suptitle(f"Variance of Number of Particles after Stabilization vs Starting Number of Particles")
    plt.title("1000 rounds of stabilization", size = 10)
    plt.savefig("Plots/Variance_of_particles.png")

    plt.figure(2)
    plt.plot(np.array(nList), np.array(np.log(listvar)))
    plt.xlabel("Number of Starting Particles")
    plt.ylabel(f"Log(Variance) of Number of Particles")
    plt.suptitle(f"Log(Variance) of Number of Particles after Stabilization vs Starting Number of Particles")
    plt.title("1000 rounds of stabiliztion", size = 10)
    plt.savefig("Plots/log_variance_of_particles.png")

    # plt.figure(3)
    # plt.plot(np.array(range(len(avgDens))), np.array(avgDens))
    # plt.plot(np.array(range(len(avgDens))), np.array(np.repeat(np.mean(np.array(avgDens), axis = 0), len(avgDens))), color = "hotpink")
    # plt.ylim(0.6, 0.9)
    # plt.xlabel("Number of Rounds (Trials)")
    # plt.ylabel(f"Average Density (# of Particles / Length of Array ({nparticles}))")
    # plt.suptitle("Particle Density of Driven-Dissipative ARW Model")
    # plt.title(f"Average density: {np.round(np.mean(np.array(avgDens), axis = 0), 3)}", size = 10)
    # plt.legend(["Density per Round", "Average After All Rounds"], loc="lower right")
    # plt.savefig(f"Plots/Average_density_{ntrials}_trials_{nparticles}_particles.png")

    # plt.figure(4)
    # plt.plot(np.array(range(len(numTopples))), np.array(numTopples))
    # plt.xlabel("Number of Rounds (Trials)")
    # plt.ylabel(f"Number of Topples to Stabilization ({nparticles}))")
    # plt.title("Number of Topples to Stabilization of Driven-Dissipative ARW Model")
    # plt.savefig(f"Plots/Number_of_Topples_{ntrials}_trials_{nparticles}_particles.png")
    # print(f"Average density: {np.mean(np.array(avgDens), axis = 0)}")





