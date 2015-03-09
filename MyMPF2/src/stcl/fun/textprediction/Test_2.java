package stcl.fun.textprediction;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.LinkedList;
import java.util.Random;
import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Brain;
import stcl.algo.predictors.VOMM;

public class Test_2 {
	private int windowLength = 1000;
	private double movingAverage;
	private int movingSum;
	private LinkedList<Integer> hits;
	private Random rand = new Random(1234);
	VOMM<Double> predictor;

	public static void main(String[] args) {
		Test_2 t = new Test_2();
		String filepath = "D:/Users/Simon/Documents/Experiments/VOMM/Book_500";
		int numUnits = 1;
		int markovOrder = 5;
		try {
			t.run(filepath, numUnits, markovOrder);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}	
	
	public void run(String filepath, int numUnits, int markovOrder) throws FileNotFoundException{
		
		predictor = new VOMM<Double>(markovOrder, 0.1);
		
		for (int i = 0; i < 10; i++){
			movingAverage = 0;
			movingSum = 0;
			hits =  new LinkedList<Integer>();
			predictor.flushMemory();
			runExperimentRound_WithSpace(filepath);
			System.out.println("Average: " + movingAverage);
		}
		
		writeSomething("the", 200);
		//System.out.println();
		
		//predictor.printTrie();
	}
	
	private void writeSomething(String startWord, int textLength){
		predictor.setLearning(false);
		predictor.flushMemory();
		
		double prediction = 0;
		for (int i = 0; i < startWord.length(); i++){
			String c = "" + startWord.substring(i, i+1);
			char character = c.charAt(0);
			double inputValue = convertCharToDouble(character);

			predictor.addSymbol(inputValue);
			prediction =  predictor.predict();		
		}
		
		char nextChar = convertDoubleToCharacter(prediction);
		for (int i = 0; i < textLength; i++){
			double inputValue = convertCharToDouble(nextChar);			
			predictor.addSymbol(inputValue);
			prediction =  predictor.predict();	
			nextChar = convertDoubleToCharacter(prediction);
			System.out.print(nextChar);

		}
		
		/*
		System.out.println();
		for (double d : entropy){
			System.out.println(d);
		}
		*/
		predictor.setLearning(true);
	}
	
	public void runExperimentRound_WithSpace(String dataFilePath) throws FileNotFoundException{
		File file = new File(dataFilePath);
		Scanner inputFile;
		inputFile = new Scanner(file);
		char predictedChar = 0;
		
		while (inputFile.hasNext()){
			Scanner curLine = new Scanner(inputFile.nextLine());
			
			String line = curLine.nextLine();
			String cleanedLine = line.replaceAll("[^\\p{L}\\p{M}\\p{P}\\p{Nd}\\s]+", ""); //Remove unwanted characters
			for (int i = 0; i < cleanedLine.length(); i++){
				String symbol = cleanedLine.substring(i, i+1);
				symbol = symbol.toLowerCase();
				
				//Get ASCII value
				char character = symbol.charAt(0);
				
				//Normalize
				double inputValue = convertCharToDouble(character);
								
				//Was prediction correct?
				int hit = 0;
				
				if (predictedChar == character) hit = 1;
				
				hits.addLast(hit);
				movingSum += hit;
				if (hits.size() > windowLength){
					int value = hits.removeFirst();
					movingSum -= value;
				}
				movingAverage = (double) movingSum / hits.size();		
				
				predictor.addSymbol(inputValue);
				Double predictionObject =  predictor.predict();
				double prediction = 0;
				if (predictionObject != null) prediction = predictionObject.doubleValue();
				
				//Convert prediction to symbol
				predictedChar = convertDoubleToCharacter(prediction);			
			}
			
			//System.out.println("Fitness: " + movingAverage);
		
			
			curLine.close();
		}
		inputFile.close();
	}
	
	private char convertDoubleToCharacter(double d){
		float f = Math.round(d * 255);
		int i = (int) f;
		char c = (char) i;
		return c;
	}
	
	private double convertCharToDouble(char c){
		int actualInt = (int) c;
		
		//Normalize
		double d = (double) actualInt / 255; //255 Is the highest ASCII value
		
		return d;
	}

}
