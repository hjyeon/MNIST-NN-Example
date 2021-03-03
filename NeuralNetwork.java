import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.lang.Math;

public class NeuralNetwork {
	
    static Double learningRate = 0.1;

    static Double[][] wih = new Double[392][785]; //layer 1 weights + bias
    static Double[] who = new Double[393]; //layer 2 weights + bias
    static Random rng = new Random();
    double[][] trainingSet;
    double[][] testingSet;
    int numTrainingSet; 


    public NeuralNetwork(String trainingSetPath, String testingSetPath) throws FileNotFoundException, IOException{
    	trainingSet = ImageParser.listToArray(ImageParser.parseTrainingRecords(trainingSetPath));
    	testingSet = ImageParser.listToArray(ImageParser.parseTestingRecords(testingSetPath));
    	numTrainingSet = trainingSet.length;
        for(int i = 0; i < wih.length; i ++){
            for (int j = 0; j < wih[0].length; j++){
                wih[i][j] = 2 * rng.nextDouble() - 1;
            }
        }
        for(int i = 0; i < who.length; i ++){
            who[i] = 2 * rng.nextDouble() - 1;
        }
    }
    
    public static double sigmoid(double x){
    	return 1.0 / (1.0 + Math.exp(-1.0 * (x)));
    }
    
    /*
     * Derivative of Sigmoid is also (1-A)(A) if we let A = 1.0 / (1.0 + Math.exp(-1.0 * (x)))
     */
    public static double derivativeSigmoid(double x){
    	return Math.exp(-1.0*(x)) / Math.pow(1.0 + Math.exp(-1.0 * (x)),2) ;
    }
    
    private void gradientDescent(int maxEpoch) {
    	
        for(int epoch = 1; epoch <= maxEpoch; epoch ++ ){
            double[] out_o = new double[numTrainingSet];
            double[][] out_h = new double[numTrainingSet][393];
            for(int i = 0; i < numTrainingSet; ++ i)
                out_h[i][392] = 1.0;

            for(int ind = 0; ind < numTrainingSet; ++ ind){
                double[] row = trainingSet[ind];
                double label = row[0]; // label for either 0 or 1. 
           //     System.out.println(row.length);

                //First layer, calculate out_h
                for(int i = 0; i < 392; ++ i) {
                    double s = 0.0;
                    for (int j = 0; j < 785; ++j) {
                        s += wih[i][j] * row[j+1]; //linear combination
                    }
                    out_h[ind][i] = NeuralNetwork.sigmoid(s); //apply sigmoid for ai
                }

                //Second layer, calculate out_o
                double s = 0.0;
                for(int i = 0; i < 393; ++ i){
                    s += out_h[ind][i] * who[i];
                }
                out_o[ind] = 1.0 / (1.0 + Math.exp(-s));

                //calc delta, consider derivative
                double[] delta = new double[393];
                for(int i = 0; i < 393; ++i){
                	delta[i] = (1-out_h[ind][i])* out_h[ind][i] * who[i] * (label - out_o[ind]);
                    //delta[i] = NeuralNetwork.derivativeSigmoid(out_h[ind][i]) * who[i] * (label - out_o[ind]);
                }

                //update the weights wih
                for(int i = 0; i < 392; ++i){
                    for(int j = 0; j < 785; ++ j){
                        wih[i][j] += learningRate * delta[i] * row[j+1];
                    }
                }

                //update who
                for(int i = 0; i < 393; ++ i){
                    who[i] += learningRate * (label - out_o[ind]) * out_h[ind][i];
                }
            }

/*
            //calc error
            double error = 0;
            for(int ind = 0; ind < numTrainingSet; ind ++){
                double[] row = trainingSet[ind];
                error += -row[0] * Math.log(out_o[ind]) - (1-row[0]) * Math.log(1- out_o[ind]);
            }
            */
/*
            //correct
            double correct = 0.0;
            for(int ind = 0; ind < numTrainingSet; ind ++){
                if ((trainingSet[ind][0] == 1.0 && out_o[ind] >=0.5) || (trainingSet[ind][0] ==0.0 && out_o[ind] < 0.5) )
                    correct += 1.0;
            }
    */
         //   double acc = correct / numTrainingSet;

           // System.out.println("Epoch: " + epoch + ", error: " + error + ", acc: " + acc);

        }
    }
    private double[] gradientDescentTest() {
    		int testingSetSize = this.testingSet.length;
            double[] out_o = new double[testingSetSize];
            double[][] out_h = new double[testingSetSize][393];
           
            
            for(int ind = 0; ind < testingSetSize ; ++ ind){
            	double[] row = this.testingSet[ind];
                //First layer, calculate out_h
                for(int i = 0; i < 392; ++ i) {
                    double s = 0.0;
                    for (int j = 0; j < 785; ++j) {
                        s += wih[i][j] * row[j]; //linear combination
                    }
                    out_h[ind][i] = NeuralNetwork.sigmoid(s); //apply sigmoid for ai
                }

                //Second layer, calculate out_o
                double s = 0.0;
                for(int i = 0; i < 393; ++ i){
                    s += out_h[ind][i] * who[i];
                }
                out_o[ind] = 1.0 / (1.0 + Math.exp(-s));
            }
            return out_o;
    }
   
    public static void main(String[] args) throws IOException {       
    	NeuralNetwork test = new NeuralNetwork("mnist_train.csv", "test3.txt");
    	test.gradientDescent(2);
    	// Print first layer weights
    	System.out.println("First Layer Weights;");
    	for (int i = 0 ; i < 785 ; i++ ) {
    		for (int j = 0 ; j < 392  ; j++ ) {
    			if (j!=391) {
    				System.out.printf("%.4f",wih[j][i]);
    				System.out.print(",");
    			}
    			else System.out.printf("%.4f",wih[j][i]);
    		}
    		System.out.print(System.lineSeparator());
    	}
    	
    	// Print second layer weights
    	System.out.println("Second Layer Weights:");
    	for (int i = 0 ; i < 393 ; i++ ) {

    		System.out.printf("%.4f",who[i]); 
    		System.out.print(",");
    	}
    	
    	//Print Testing result

    	double[] result = test.gradientDescentTest();
    	System.out.println("test set size: " + result.length);
    	System.out.println(System.lineSeparator()+"Result");
    	for (double s : result) {
    	System.out.printf("%.2f", s);
    	System.out.print(",");
    	}
    	System.out.println(System.lineSeparator()+"Results rounded");
    	for (double s : result) {
    	System.out.print(Math.round(s));
    	System.out.print(",");
    	}
    }
}
