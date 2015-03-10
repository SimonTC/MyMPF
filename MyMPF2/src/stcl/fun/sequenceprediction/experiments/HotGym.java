package stcl.fun.sequenceprediction.experiments;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Random;
import java.util.Scanner;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Brain;
import stcl.algo.brain.NU;
import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.util.Normalizer;
import stcl.fun.sequenceprediction.CopyOfSequenceTrainer;
import stcl.fun.sequenceprediction.SequenceTrainer;

public class HotGym {
	private double[] data;
	private ArrayList<Double> output;
	private ArrayList<Double> xValues;
	private JFrame frame;
	
	private final int ITERATIONS = 10;
	boolean sin = false;
	SimpleMatrix uniformDistribution;
	
	public static void main(String[] args){
		HotGym runner = new HotGym();
		
		runner.start();
	} 
	
	public void start(){
		//Load data
		String dataFilePath = "D:/Users/Simon/Documents/Experiments/Hotgym/data_normalized_3000.csv";
		int iterations = ITERATIONS;
				
		try {
			if (sin){
				data = createData(1000);
			} else {
				data= loadData(dataFilePath);
			}
					
			//Create neocortical unit
			Brain brain = createUnit(iterations);
					
			
			//Do test
			ArrayList<double[]> list = new ArrayList<double[]>();
			list.add(data);
			Random rand = new Random();
			CopyOfSequenceTrainer trainer = new CopyOfSequenceTrainer(list, ITERATIONS, rand );
			boolean calculateErrorAsDistance = true;
			ArrayList<Double> errors = trainer.train(brain, 0, calculateErrorAsDistance, null);
			
			for ( double d : errors) System.out.println(d);
					
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void test(int iterations, NU unit, ArrayList<Double> data){
		SimpleMatrix ffOutput;
		SimpleMatrix fbOutput;
		double error;
		
		for (int iteration = 0; iteration < iterations; iteration++){
			float start = System.nanoTime();
			output = new ArrayList<Double>();
			xValues = new ArrayList<Double>();
			output.add((double) 0);
			double mse = 0;
			unit.flush();
			for (int i = 0; i < data.size() - 1; i++){
				xValues.add((double) (i+1)); 
				double input = data.get(i);
				double[][] inputDataVector = {{input}};
				SimpleMatrix inputVector = new SimpleMatrix(inputDataVector);
				double expectedOutput = data.get(i+1);
				ffOutput = unit.feedForward(inputVector);
				fbOutput = unit.feedBackward(uniformDistribution);
				
				double actualOutput = fbOutput.get(0);
				
				output.add(actualOutput);
				
				error = actualOutput - expectedOutput;;
				error *= error;
				mse += error;
				
				/*
				System.out.printf("Exp: %1$.2f Act: %1$.2f", expectedOutput, actualOutput);
				System.out.println();
				*/
				
				
				
				
			}	
			float end = System.nanoTime();
			float duration = end - start;
			float duration_seconds = duration / 1000000000;
			
			
			
			mse *= (double) 1/data.size();
			System.out.printf("Iteration: " + iteration + " MSE: %1$.4f" , mse);
			System.out.println(" Duration: " + duration_seconds + " seconds");
			System.out.println();
			
			
		}
	}
	
	
	
	private Brain createUnit(int maxIterations){
		Random rand = new Random();
		int ffInputLength = 1;
		int spatialMapSize = 10;
		int temporalMapSize = 10;
		int markovOrder = 1;
		
		Brain brain = new Brain(1, rand, ffInputLength, spatialMapSize, temporalMapSize, markovOrder);

		return brain;
	}
	
	private double[] loadData(String filePath) throws FileNotFoundException{
		ArrayList<Double> arr = new ArrayList<Double>();
		File file = new File(filePath);
		Scanner inputFile;
		inputFile = new Scanner(file);
		
		while (inputFile.hasNext()){
			Scanner curLine = new Scanner(inputFile.nextLine());
			curLine.useLocale(Locale.US); //Has to use US locale to make sure that '.' is used as decimal delimiter
			//Jump the id
			String s = curLine.next();			
			//Read value
			String line[] = s.split(";");
			double val = Double.parseDouble(line[1]);
			arr.add(val);
			
			curLine.close();
		}
		inputFile.close();
		
		double[] d = new double[arr.size()];
		for (int i = 0; i < arr.size(); i++){
			d[i] = arr.get(i);
		}
		
		return d;
	}
	
	private double[] createData(int samples) {
		double[] arr = new double[samples];
		
		for (int i = 0; i < samples; i++){
			double d = Math.sin((double) i / 10);
			arr[i] = d;
		}
		
		return arr;
	}

}
