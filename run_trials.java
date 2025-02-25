import java.util.*;

public class run_trials{

    public static void runBasicTrial(int nTrials, int nParticles, double lambda_rate){
        arw_DD objTemp = new arw_DD();

        List<List<Double>> endRound = objTemp.runNTrials(nParticles, nTrials, lambda_rate);

        System.out.println("avgDens: " + endRound.get(0));
        System.out.println("numTopples: " + endRound.get(1));
    }

    public static List<Long> timeTrials(List<Integer> nPar_Try, int nTrials, double lambda_rate){
        List<Long> times = new ArrayList<>();

        for(int n: nPar_Try){
            List<Long> roundTimes = new ArrayList<>();
            for (int i  = 0; i < 10; i++){
                long startTime = System.currentTimeMillis();
                runBasicTrial(nTrials, n, lambda_rate);
                long endTime = System.currentTimeMillis();
                long time = endTime - startTime;
                roundTimes.add(time);
            }
            //System.out.println(roundTimes);
            long avgTime = calculateAverage(roundTimes);
            times.add(avgTime);
        }

        return times;
    }
    public static void main(String[] args){
        //runBasicTrial(100, 100, 0.5);

        List<Integer> nPar_Try = new ArrayList<>(Arrays.asList(1, 2, 4, 8, 10));
        List<Long> times = timeTrials(nPar_Try, 10, 0.5);
        System.out.println(times);
    }

    private static long calculateAverage(List <Long> counts) {
        long sum = 0;
        if(!counts.isEmpty()) {
          for (long count : counts) {
              sum += count;
          }
          return sum / counts.size();
        }
        return sum;
      }
}
