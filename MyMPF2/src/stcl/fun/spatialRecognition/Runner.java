package stcl.fun.spatialRecognition;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SpatialPooler;
import stcl.algo.som.SomNode;

public class Runner {

	public static void main(String[] args) {
		Runner runner = new Runner();
		//runner.run();
		runner.bigRun();

	}
	
	public void run(){
		int iterations = 200;
		int uniqueSamples = 90;
		int sampleLength = 5;
		int mapsize = 10;
		Random rand = new Random();
		
		//Create data
		double[][] data = createData(uniqueSamples, sampleLength, rand);		
		
		//Create pooler
		SpatialPooler pooler = new SpatialPooler(rand, iterations, sampleLength, mapsize);
		
		//Train pooler
		train(pooler, data, iterations);
		
		//Evaluate pooler
		evaluate(pooler, data);
	}
	
	public void bigRun(){
		int maxIterations = 200;
		int iterationStep = 10;
		
		int maxUniqueSamples = 50;
		int uniqueSampleStep = 5;
		
		int maxSampleLength = 10;
		int sampleLengthStep = 1;
		
		int maxMapsize = 20;
		int mapSizeStep = 2;
		
		Random rand = new Random();
		
		double[][] meanSquaredErrors = new double[(maxIterations/iterationStep) * (maxUniqueSamples/uniqueSampleStep) * (maxSampleLength/sampleLengthStep) * (maxMapsize/mapSizeStep)][5];
		
		int counter = 0;
		
		for (int iterations = iterationStep; iterations <= maxIterations; iterations+=iterationStep){
			for (int uniqueSamples = uniqueSampleStep; uniqueSamples <= maxUniqueSamples; uniqueSamples +=uniqueSampleStep){
				for (int sampleLength = sampleLengthStep; sampleLength <= maxSampleLength; sampleLength+= sampleLengthStep){
					for (int mapSize = mapSizeStep; mapSize <= maxMapsize; mapSize+=mapSizeStep){
						double mse = 0;
						for (int i = 0; i < 5; i++){
							SpatialPooler pooler = new SpatialPooler(rand, iterations, sampleLength, mapSize);
							double[][] data = createData(uniqueSamples, sampleLength, rand);
							train(pooler, data, iterations);
							mse += evaluate(pooler, data);
						}
						mse = mse / 5;
						meanSquaredErrors[counter][0] = iterations;
						meanSquaredErrors[counter][1] = uniqueSamples;
						meanSquaredErrors[counter][2] = sampleLength;
						meanSquaredErrors[counter][3] = mapSize;
						meanSquaredErrors[counter][4] = mse;
						counter++;
						
						if (counter % 100 == 0){
							System.out.println("Sample: " + counter);
						}
					}
				}
			}
		}
		
		//Save data to file
		String filePath = "data.csv";
		saveToFile(meanSquaredErrors, filePath);
		
		System.out.println("Finished");
		
	}

	
	private void saveToFile(double[][] data, String filePath){
		File genomeFile = new File(filePath);
		try {
			PrintWriter output = new PrintWriter(genomeFile);
			for (int i = 0; i < data.length; i++){
				String s = "";
				for (int j = 0; j < data[i].length; j++){
					s+= data[i][j] + ";";
				}
				output.println(s);
			}
			
			output.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	private double[][] createData(int uniqueSamples, int sampleLength, Random rand){
		double[][] data = new double[uniqueSamples][sampleLength];
		
		for (int i = 0; i < uniqueSamples; i++){
			for (int j = 0; j < sampleLength; j++){
				data[i][j] = rand.nextDouble();
			}
		}
		
		return data;
	}
	
	private void train(SpatialPooler pooler, double[][] data, int iterations){
		for (int i = 0; i < iterations; i++){
			for (int j = 0; j < data.length; j++){
				double[][] input = {data[j]};
				SimpleMatrix inputVector = new SimpleMatrix(input);
				pooler.feedForward(inputVector);
			}
		}
	}
	
	private double evaluate(SpatialPooler pooler, double[][] data){
		double mse = 0;
		for (int j = 0; j < data.length; j++){
			double[][] input = {data[j]};
			SimpleMatrix inputVector = new SimpleMatrix(input);
			SomNode inputNode = new SomNode(inputVector);
			SomNode bmu = pooler.getSOM().getBMU(inputVector);
			double[] output = bmu.getVector().getMatrix().data;
			double error = bmu.squaredDifference(inputVector);
			/*
			System.out.print("Expected: ");
			printArray(data[j]);
			System.out.println();
			
			
			System.out.print("Actual:   ");
			printArray(output);
			System.out.println();
			
			System.out.printf("Error:    %1$.3f  ", error);
			System.out.println();
			System.out.println();
			*/
			mse+= error;
		}	
		
		mse = ((double)mse / data.length);
		/*
		System.out.printf("Mean squared error: %1$.3f  ", mse);
		*/
		
		return mse;
		
	}
	
	
	private void printArray(double[] array){
		String s = "";
		for (int i = 0; i < array.length; i++){
			System.out.printf("%1$.3f  ", array[i]);
		}
	}
	

}
