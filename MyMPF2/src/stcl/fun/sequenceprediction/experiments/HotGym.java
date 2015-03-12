package stcl.fun.sequenceprediction.experiments;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Random;
import java.util.Scanner;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Brain;
import stcl.algo.brain.Brain_DataCollector;
import stcl.algo.util.FileWriter;
import stcl.fun.sequenceprediction.SequenceTrainer;

public class HotGym {
	private double[] data;
	private ArrayList<Double> output;
	private ArrayList<Double> xValues;
	private JFrame frame;
	private Brain_DataCollector brain;
	
	private final int ITERATIONS = 5;
	
	SimpleMatrix uniformDistribution;
	
	boolean sin = false;
	boolean writeOutputsToFile = false;
	
	public static void main(String[] args){
		HotGym runner = new HotGym();
		
		runner.start();
	} 
	
	public void start(){
		//Load data
		String dataFilePath = "D:/Users/Simon/Documents/Experiments/Hotgym/data_normalized_3000.csv";
		//String dataFilePath = "c:/Users/Simon/Documents/Experiments/OMXC20/OMXC20_Normalized.csv";
		int iterations = ITERATIONS;
				
		try {
			if (sin){
				data = createData(1000);
			} else {
				data= loadData(dataFilePath);
			}
					
			//Create neocortical unit
			brain = createBrain(iterations);
			brain.setCollectData(writeOutputsToFile);
			
								
			//Do test
			ArrayList<double[]> list = new ArrayList<double[]>();
			list.add(data);
			Random rand = new Random();
			SequenceTrainer trainer = new SequenceTrainer(list, ITERATIONS, rand );
			boolean calculateErrorAsDistance = true;
			ArrayList<Double> errors = trainer.train(brain, 0, calculateErrorAsDistance);
			
			for ( double d : errors) System.out.println(d);
			
			if (writeOutputsToFile){
				String datafolder = "c:/Users/Simon/Documents/Experiments/OMXC20";
				writeOutputsToFile(datafolder);
				System.out.println("Outputs written to " + datafolder);
			}
					
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void writeOutputsToFile(String dataFolder){
		FileWriter writer = new FileWriter();
		try {
			writer.openFile(dataFolder + "/Predictions.csv", false);
			//Collect output data
			ArrayList<SimpleMatrix> outputs = brain.getReturnedOutputs();
			for (SimpleMatrix m : outputs){
				double d = m.get(0);
				writer.writeLine("" + d);
			}
			writer.closeFile();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private Brain_DataCollector createBrain(int maxIterations){
		Random rand = new Random();
		int ffInputLength = 1;
		int spatialMapSize = 10;
		int temporalMapSize = 10;
		int markovOrder = 2;
		
		Brain_DataCollector brain = new Brain_DataCollector(1, rand, ffInputLength, spatialMapSize, temporalMapSize, markovOrder);

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
