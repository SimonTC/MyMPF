package stcl.fun.hotgym;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Random;
import java.util.Scanner;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;
import org.math.plot.Plot2DPanel;

import stcl.algo.brain.NeoCorticalUnit;

public class Runner {
	private ArrayList<Double> data;
	private ArrayList<Double> output;
	private ArrayList<Double> xValues;
	
	private final int ITERATIONS = 100;
	private final boolean VISUALIZE = true;
	
	public static void main(String[] args){
		Runner runner = new Runner();
		
		runner.start();
	} 
	
	public void start(){
		//Load data
				String dataFilePath = "D:/Users/Simon/Documents/Experiments/Hotgym/data_normalized_3000.csv";
				int iterations = ITERATIONS;
				
				boolean sin = true;
				try {
					if (sin){
						data = createData(1000);
					} else {
						data= loadData(dataFilePath);
					}
					
					//Create neocortical unit
					NeoCorticalUnit nu = createUnit(iterations);
					
					//Do test
					test(iterations, nu, data);
					
				} catch (FileNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
	}
	
	private void test(int iterations, NeoCorticalUnit unit, ArrayList<Double> data){
		SimpleMatrix ffOutput;
		SimpleMatrix fbOutput;
		double error;
		
		for (int iteration = 0; iteration < iterations; iteration++){
			output = new ArrayList<Double>();
			xValues = new ArrayList<Double>();
			output.add((double) 0);
			double mse = 0;
			unit.resetTemporalDifferences();
			for (int i = 0; i < data.size() - 1; i++){
				xValues.add((double) (i+1));
				double input = data.get(i);
				double[][] inputDataVector = {{input}};
				SimpleMatrix inputVector = new SimpleMatrix(inputDataVector);
				double expectedOutput = data.get(i+1);
				ffOutput = unit.feedForward(inputVector);
				fbOutput = unit.feedBackward(ffOutput);
				
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
			
			//Plot data
			if (VISUALIZE) plot(iteration);
			
			
			mse *= (double) 1/data.size();
			System.out.printf("Iteration: " + iteration + " MSE: %1$.4f" , mse);
			System.out.println();
			
			
		}
	}
	
	private void plot(int iteration){
		double[] x = toArray(xValues);
		
		double[] y_expected = toArray(data);
		
		double[] y_actual = toArray(output);
				 
				  // create your PlotPanel (you can use it as a JPanel)
				  Plot2DPanel plot = new Plot2DPanel();
				 
				  // add a line plot to the PlotPanel
				  plot.addLinePlot("Expected", x, y_expected);
				  plot.addLinePlot("Actual", x, y_actual);
				 
				  // put the PlotPanel in a JFrame, as a JPanel
				  JFrame frame = new JFrame("Iteration " + iteration);
	                frame.setSize(600, 600);
	                frame.setContentPane(plot);
	                frame.setVisible(true);
				 
	}
	
	private double[] toArray(ArrayList<Double> list){
		double[] arr = new double[list.size()];
		
		for (int i = 0; i < arr.length; i++){
			arr[i] = list.get(i);
		}
		return arr;
	}
	
	private NeoCorticalUnit createUnit(int maxIterations){
		Random rand = new Random();
		int ffInputLength = 1;
		int spatialMapSize = 10;
		int temporalMapSize = 10;
		double initialPredictionLearningRate = 0.8;
		boolean useMarkovPrediction = true;
		double leakyCoefficient = 0.4;		
		
		NeoCorticalUnit nu = new NeoCorticalUnit(rand, maxIterations, ffInputLength, spatialMapSize, temporalMapSize, initialPredictionLearningRate, useMarkovPrediction, leakyCoefficient);
		
		return nu;
	}
	
	private ArrayList<Double> loadData(String filePath) throws FileNotFoundException{
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
		
		return arr;
	}
	
	private ArrayList<Double> createData(int samples) throws FileNotFoundException{
		ArrayList<Double> arr = new ArrayList<Double>();
		
		for (int i = 0; i < samples; i++){
			double d = Math.sin((double) i / 10);
			arr.add(d);
		}
		
		return arr;
	}

}
