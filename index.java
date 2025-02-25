public class index {
    private String state;
    private int numPar;

    public index(String state, int numPar){
        this.state = state;
        this.numPar = numPar;
    }

    public index(String state){
        this(state, 0);
    }

    public index(int numPar){
        this("A", numPar);
    }

    public index(){
        this("A", 0);
    }

    public void editParticle(int change){
        this.numPar += change;
    }

    public void editState(String state){
        this.state = state;
    }

    public int getParticle(){
        return this.numPar;
    }

    public String getState(){
        return this.state;
    }

    public String toString(){
        return "State: " + this.state + ", numPar = " + this.numPar;
    }
}
