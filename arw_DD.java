import java.util.*;

public class arw_DD {
    public List<List<Double>> runNTrials(int nParticles, int nTrials, double lambda_rate){
        Random r = new Random();
        List<index> arr = setUp(nParticles);
        List<Double> avgDens = new ArrayList<>();
        List<Double> numTopples = new ArrayList<>();

        List<Integer> probs = generateDistribution(lambda_rate);

        for (int i = 0; i < nTrials; i++){
            returnTwo pair = runRound(arr, (arr.size() / 2), arr.size(), probs, r);
            avgDens.add(pair.getDens());
            numTopples.add((double) pair.getTopples());
        }

        List<List<Double>> returnVals = new ArrayList<>();
        returnVals.add(avgDens);
        returnVals.add(numTopples);

        return returnVals;
    }

    private List<index> setUp(int n){
        List<index> arr = new ArrayList<>();
        for(int i = 0; i < n; i++){
            arr.add(new index(1));
        }

        return arr;
    }

    private returnTwo runRound(List<index> arr, int addIndex, int lenArr, List<Integer> probs, Random r){
        boolean topple = true;
        int numTopples = 0;

        List<Integer> instr = getInstr(probs, 100*arr.size(), r);
        arr.get(addIndex).editParticle(1);
        arr.get(addIndex).editState("A");

        while(topple){
            int roundTopples = 0;
            for(int i = 0; i < lenArr; i++){
                index currIndex = arr.get(i);
                if(currIndex.getState().equals("A") && currIndex.getParticle() > 0){
                    if(instr.size() < 1){
                        instr = getInstr(probs, 100*arr.size(), r);
                    }

                    int currInstr = instr.remove(instr.size() - 1);
                    roundTopples += 1;

                    if(currInstr == 0 & currIndex.getParticle() == 1){
                        currIndex.editState("S");
                        numTopples += 1;
                        //roundTopples += 1;
                    } else if(currInstr == -1 || currInstr == 1){
                        currIndex.editParticle(-1);

                        if(currIndex.getParticle() < 1){
                            currIndex.editState("S");
                        } else{
                            currIndex.editState("A");
                        }

                        if((i + currInstr) < lenArr && (i + currInstr) >= 0){
                            arr.get(i + currInstr).editParticle(1);
                            arr.get(i + currInstr).editState("A");
                        }

                        numTopples += 1;
                        //roundTopples += 1;
                    }
                }

                topple = roundTopples > 0 ? true: false;
            }
        }


        int numPar = 0;
        for(int i = 0; i < arr.size(); i++){
            numPar += arr.get(i).getParticle();
        }
        System.out.println("Average Density: " + ((double) (numPar) / arr.size()));
        returnTwo returnVal = new returnTwo(((numPar * 1.0) / arr.size()), numTopples);
        return returnVal;
    }

    private List<Integer> generateDistribution(double rate){
        double left =  100 / (2 *(1 + rate));
        double sleep = (rate * 100) / (1+rate);

        List<Integer> possNum = new ArrayList<>();

        for(int i = 0; i < (int) left; i++){
            possNum.add(-1);
            possNum.add(1);
        }

        for(int i = 0; i < (int) sleep; i++){
            possNum.add(0);
        }

        return possNum;
    }

    private List<Integer> getInstr(List<Integer> probs, int n, Random r){
        List<Integer> instr = new ArrayList<>(n);
        for(int i = 0; i < n; i++){
            instr.add(i, probs.get(r.nextInt(probs.size())));
        }

        return instr;
    }



    private class returnTwo{
        private double dens;
        private int numTopples;

        private returnTwo(double dens, int numTopples){
            this.dens = dens;
            this.numTopples = numTopples;
        }

        public double getDens(){
            return this.dens;
        }

        public int getTopples(){
            return this.numTopples;
        }
    }    
}


