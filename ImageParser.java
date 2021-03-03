import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.lang.Math;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.MatrixType;
import org.ejml.ops.ConvertDMatrixStruct;
import org.ejml.ops.MatrixIO;
import org.ejml.simple.SimpleMatrix;

public class ImageParser {
    static String firstDigit = "3"; // Label 0
    static String secondDigit = "0"; // Label 1

    
    public static ArrayList<double[]> parseTrainingRecords(String file_path) throws FileNotFoundException, IOException {
    	ArrayList<double[]> data = new ArrayList<double[]>();
        BufferedReader reader = new BufferedReader(new FileReader(file_path));
        String line = "";
        int numCol = 0;
        while ((line = reader.readLine()) != null) {
            String[] string_values = line.split(",");
            numCol = string_values.length;
            if (!string_values[0].equals(firstDigit) && !string_values[0].equals(secondDigit)) 
            	continue;
            double[] double_values = new double[numCol];
            if (firstDigit.equals(string_values[0])) double_values[0] = 0.0; // label 0
            else double_values[0] = 1.0; // label 1
            for (int i = 1 ; i < numCol ; i++) {
                double_values[i] = Double.parseDouble(string_values[i])/255.0; // features
            }
            data.add(double_values);
        }
        reader.close();
    return data;
    }
    
    public static ArrayList<double[]> parseTestingRecords(String file_path) throws FileNotFoundException, IOException {
    	ArrayList<double[]> data = new ArrayList<double[]>();
        BufferedReader reader = new BufferedReader(new FileReader(file_path));
        String line = "";
        int numCol = 0;
        while ((line = reader.readLine()) != null) {
            String[] string_values = line.split(",");
            numCol = string_values.length;
            double[] double_values = new double[numCol];
            for (int i = 1 ; i < numCol ; i++) {
                double_values[i] = Double.parseDouble(string_values[i])/255.0; // features
            }
            data.add(double_values);
        }
        reader.close();
    return data;
    }
    
    public static double[][] listToArray (ArrayList<double[]> list){
    	int numRow = list.size();
    	int numCol = list.get(0).length;
    	double[][] array = new double[numRow][numCol+1];
    	for (int i = 0 ; i < numRow ; i ++) {
    		for (int j=0 ; j < numCol ; j++ ) {
        		array[i][j] = list.get(i)[j];
    		}
    		array[i][numCol] = 1.0;

    	}
    	return array;
    }
    
    
    
    
    public static void main(String[] args) throws IOException {
        // Parse csv files
    	
        //double[][] records = listToArray(parseTrianingRecords("mnist_train.csv"));
        double[][] test_records = listToArray(parseTestingRecords("minitest.txt"));
        DMatrixRMaj test  = new DMatrixRMaj(test_records);
        MatrixIO.saveDenseCSV(test, "minitest_file.csv");
        


    }


}
