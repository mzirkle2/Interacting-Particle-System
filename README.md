# Interacting-Particle-System

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
        avgDens, numTopples = runNTrials(n, 1000)
        listParticles.append([i * n for i in avgDens]) ## one entry in list is list of number of particles at end of each stabilization round

    listParticles = np.var(listParticles, axis = 1) ## final list has one entry as variance of the list of number of particles at end of each stablization round
    return listParticles
