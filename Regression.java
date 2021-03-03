import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.ejml.data.DMatrixRMaj;
import org.ejml.ops.MatrixIO;

public class Regression {
    private Random rng = new Random();
    private double b; // bias
    private double[] w; // Weights, will be size 784
    private Double[] as; // ai's
    private double prevCost;
    private double newCost;
    private double learning_rate = 0.001;
    List<double[]> trainingSet;
    List<double[]> testingSet;
    private final int imageSize = 784; // 28 * 28
    
    public Regression(String trainingSetPath, String testingSetPath) throws FileNotFoundException, IOException{

    	trainingSet = ImageParser.parseTrainingRecords(trainingSetPath);
    	testingSet = ImageParser.parseTestingRecords(testingSetPath);
    	b = rng.nextDouble(); // initial bias
    	w = new double[imageSize];
    	for (int i = 0; i < w.length; i++) w[i] = rng.nextDouble(); // initial weight
    	prevCost = 0.0;
    	newCost = 0.0;
    	
    }
    
    private void gradientDescent(int maxNumIterations) {
        // Gradient descent step
        for (int iteration = 1; ; iteration++) {
 
        	this.learning_rate = learning_rate/Math.sqrt(1.5*iteration); //decreasing learning rate

            // Calculate a_i array
            Double[] a = new Double[this.trainingSet.size()];
            for (int i = 0; i < this.trainingSet.size(); i++) {
                double sum_wx = 0;
                for (int j = 0; j < w.length; j++) {
                    sum_wx += w[j] * this.trainingSet.get(i)[j+1]; //1st index is the label 0 or 1
                }
                a[i] = 1.0 / (1.0 + Math.exp(-1.0 * (sum_wx + b)));
            }

            // Update weights and bias
            for (int j = 0; j < w.length; j++) {
                double w_temp = 0;
                for (int i = 0; i < this.trainingSet.size(); i++) {
                    w_temp += (a[i] - this.trainingSet.get(i)[0]) * this.trainingSet.get(i)[j+1];
                }
                w[j] = w[j] - learning_rate * w_temp; // update weights
            }

            // Update bias
            double b_temp = 0;
            for (int i = 0; i < trainingSet.size(); i++) {
                b_temp += (a[i] - trainingSet.get(i)[0]);
            }
            b -= learning_rate * b_temp;

            // Calculate cost function
            prevCost = newCost;
            newCost = 0.0;
            for (int i = 0; i < trainingSet.size(); i++) {
                if (trainingSet.get(i)[0] == 0.0) {
                    if (a[i] > 0.99999) newCost += 1000.0; // something large
                    else newCost -= Math.log(1 - a[i]);
                }
                else if (trainingSet.get(i)[0] == 1.0) {
                    if (a[i] < 0.00001) newCost += 1000.0;
                    else newCost -= Math.log(a[i]);
                }
            }

            // Check for convergence
            double convergence = Math.abs(newCost - this.prevCost);
            if (convergence < 0.000000001) break;
            else if (iteration > maxNumIterations) { // termination condition
                System.out.println("Reached the maximum number of iterations. "
                        + "Maybe try a different learning rate?");
                break;
            }
           // System.out.println(convergence);
        }
    	
    }
    private void gradientDescentTest() {
    	this.as = new Double[this.testingSet.size()];
        for (int i = 0; i < this.testingSet.size(); i++) {
            double sum_wx = 0;
            for (int j = 0; j < w.length; j++) {
                sum_wx += w[j] * this.testingSet.get(i)[j]; 
            }
            as[i] = 1.0 / (1.0 + Math.exp(-1.0 * (sum_wx + b)));
        }
    }

    public static void main(String[] args) throws IOException {       
    	Regression test = new Regression("mnist_train.csv", "test3.txt");
    	test.gradientDescent(1000);
    	test.gradientDescentTest();
    	
    	System.out.println("Weights + Bias :");
    	for (int i = 0 ; i < test.w.length ; i++) {
    		System.out.print(test.w[i]+ ",");
    	}
    		System.out.print(test.b);
    	    	System.out.println();
    	
    	
    	System.out.println("Test Results: ");
    	for (int i = 0 ; i < test.as.length ; i++) {
    		System.out.printf("%.2f", test.as[i]);
    		System.out.print(",");
    	}
    	System.out.println();
    	
    	for (int i = 0 ; i < test.as.length ; i++) {
    		System.out.print(Math.round(test.as[i]));
    		System.out.print(",");
    	}
    	double sum = 0;    	
    	System.out.println(System.lineSeparator()+"Accuracy: ");
    	for (int i = 0 ; i < 100 ; i++) {
    		sum = sum + Math.round(test.as[i]);
    	//	System.out.print(",");
    	}
    	for (int i = 100 ; i < 200 ; i++) {
    		sum = sum + 1-Math.round(test.as[i]);
    	//	System.out.print(",");
    	}
    	System.out.println(100-sum/2.0+"%");
    	
    	
    	
    	
    	
    	
    	
    }

    
     

}
