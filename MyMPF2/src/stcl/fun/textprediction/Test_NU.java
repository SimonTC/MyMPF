package stcl.fun.textprediction;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.LinkedList;
import java.util.Random;
import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Brain;
import stcl.algo.util.FileWriter;

public class Test_NU {
	private int windowLength = 1000;
	private double movingAverage;
	private int movingSum;
	private LinkedList<Integer> hits;
	private Random rand = new Random(1234);
	private Brain brain;

	public static void main(String[] args) {
		Test_NU t = new Test_NU();
		String filepath = "D:/Users/Simon/Documents/Experiments/VOMM/Book_500";
		int numUnits = 1;
		int markovOrder = 5;
		try {
			t.run(filepath, numUnits, markovOrder);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}	
	
	public void run(String filepath, int numUnits, int markovOrder) throws IOException{
		
		brain = new Brain(numUnits, rand, 1, 10, 4, markovOrder);
		FileWriter writer = new FileWriter();
		writer.openFile(filepath + "_log", false);
		writer.closeFile();
		for (int i = 0; i < 10; i++){
			writer.openFile(filepath + "_log", true);
			movingAverage = 0;
			movingSum = 0;
			hits =  new LinkedList<Integer>();
			brain.flush();
			runExperimentRound_WithSpace(filepath, writer);
			System.out.println("Average: " + movingAverage);
			writer.closeFile();
		}
		
		writer.closeFile();
		
		writeSomething("the", 200);
		//System.out.println();
		
		//predictor.printTrie();
	}
	
	private void writeSomething(String startWord, int textLength){
		brain.setLearning(false);
		brain.flush();
		
		SimpleMatrix prediction = null;
		for (int i = 0; i < startWord.length(); i++){
			String c = "" + startWord.substring(i, i+1);
			char character = c.charAt(0);
			double inputValue = convertCharToDouble(character);
			
			double[][] inputData = {{inputValue}};
			SimpleMatrix inputVector = new SimpleMatrix(inputData);
			prediction = brain.step(inputVector);			
		}
		
		char nextChar = convertDoubleToCharacter(prediction.get(0));
		for (int i = 0; i < textLength; i++){
			double inputValue = convertCharToDouble(nextChar);			
			double[][] inputData = {{inputValue}};
			SimpleMatrix inputVector = new SimpleMatrix(inputData);			
			prediction = brain.step(inputVector);			
			nextChar = convertDoubleToCharacter(prediction.get(0));
			System.out.print(nextChar);

		}
		
		/*
		System.out.println();
		for (double d : entropy){
			System.out.println(d);
		}
		*/
		brain.setLearning(true);
	}
	
	public void runExperimentRound_WithSpace(String dataFilePath, FileWriter writer) throws FileNotFoundException{
		File file = new File(dataFilePath);
		Scanner inputFile;
		inputFile = new Scanner(file);
		char predictedChar = 0;
		
		try {
			writer.writeLine("");
			writer.writeLine("Input;Prediction;Entropies");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		while (inputFile.hasNext()){
			Scanner curLine = new Scanner(inputFile.nextLine());
			
			String line = curLine.nextLine();
			String cleanedLine = cleanString(line);
			for (int i = 0; i < cleanedLine.length(); i++){
				//Get ASCII value
				char character = cleanedLine.charAt(i);
				
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
				
				double[][] inputData = {{inputValue}};
				SimpleMatrix inputVector = new SimpleMatrix(inputData);
				SimpleMatrix prediction = brain.step(inputVector);
				
				//Convert prediction to symbol
				predictedChar = convertDoubleToCharacter(prediction.get(0));		
				
				writeInfo(writer, brain, character, predictedChar);
			}
			
			//System.out.println("Fitness: " + movingAverage);
		
			
			curLine.close();
		}
		inputFile.close();
	}
	
	private String cleanString(String string){
		String cleanedString = string.toLowerCase();
		cleanedString = cleanedString.replaceAll("[^a-z\\s]", ""); //Removes everything but small letters and space
		cleanedString = cleanedString.replaceAll("\\s{2,}", " "); //Replaces all double space with single space
		return cleanedString;
	}
	
	private void writeInfo(FileWriter writer, Brain brain, char input, char prediction){
		double[] entropies = brain.getEntropies();
		String line = "";
		line += input + ";";
		line += prediction + ";";
		for (double d : entropies){
			line += d + ";";
 		}
		try {
			writer.writeLine(line);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
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
