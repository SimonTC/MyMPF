package stcl.fun.textprediction;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.LinkedList;
import java.util.Scanner;

import stcl.algo.predictors.VOMM;

public class Test {
	VOMM<String> predictor;
	int windowLength = 1000;
	double movingAverage;
	int movingSum;
	LinkedList<Integer> hits;

	public static void main(String[] args) {
		Test t = new Test();
		String filepath = "C:/Users/Simon/Documents/Experiments/VOMM/Book_500";
		try {
			t.run(filepath);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}	
	
	public void run(String filepath) throws FileNotFoundException{
		predictor = new VOMM<String>(4, 0.5);
		movingAverage = 0;
		hits =  new LinkedList<Integer>();
		for (int i = 0; i < 1; i++){
			runExperimentRound_WithSpace(filepath);
		}
		
		writeSomething("the", 100);
		//System.out.println();
		
		//predictor.printTrie();
	}
	
	private void writeSomething(String startWord, int textLength){
		predictor.setLearning(false);
		predictor.flushMemory();
		
		
		for (int i = 0; i < startWord.length(); i++){
			String c = "" + startWord.substring(i, i+1);
			predictor.addSymbol(c);
		}
		
		String nextSymbol = "";
		double[] entropy = new double[textLength];
		for (int i = 0; i < textLength; i++){
			nextSymbol = predictor.predict();
			predictor.addSymbol(nextSymbol);
			System.out.print(nextSymbol);
			entropy[i] = predictor.calculateEntropy();
		}
		System.out.println();
		for (double d : entropy){
			System.out.println(d);
		}
		
		predictor.setLearning(true);
	}
	
	
	/*
	public void runExperimentRound(String dataFilePath) throws FileNotFoundException{
		File file = new File(dataFilePath);
		Scanner inputFile;
		inputFile = new Scanner(file);
		String predictedSymbol = null;
		
		while (inputFile.hasNext()){
			Scanner curLine = new Scanner(inputFile.nextLine());
			
			String line = curLine.next();
			String words[] = line.split(" ");
			for (String word : words){
				String[] symbols = word.split("");
				for (String symbol : symbols){
					//Was prediction correct?
					symbol = symbol.toLowerCase();
					if (predictedSymbol != null){
						int hit = 0;
						if (predictedSymbol.equals(symbol)) hit = 1;
						
						hits.addLast(hit);
						movingSum += hit;
						if (hits.size() > windowLength){
							int value = hits.removeFirst();
							movingSum -= value;
						}
						movingAverage = (double) movingSum / hits.size();
						
						//System.out.println("Fitness: " + movingAverage);
					}		
					//Feed symbol to model
					predictor.addSymbol(symbol);;
					//Predict
					predictedSymbol = predictor.predict(symbol);
				}
			}
		
			
			curLine.close();
		}
		inputFile.close();
	}
	*/
	public void runExperimentRound_WithSpace(String dataFilePath) throws FileNotFoundException{
		File file = new File(dataFilePath);
		Scanner inputFile;
		inputFile = new Scanner(file);
		String predictedSymbol = null;
		
		while (inputFile.hasNext()){
			Scanner curLine = new Scanner(inputFile.nextLine());
			
			String line = curLine.nextLine();
			for (int i = 0; i < line.length(); i++){
				String symbol = line.substring(i, i+1);
				symbol = symbol.toLowerCase();
				//Was prediction correct?
				if (predictedSymbol != null){
					int hit = 0;
					if (predictedSymbol.equals(symbol)) hit = 1;
					
					hits.addLast(hit);
					movingSum += hit;
					if (hits.size() > windowLength){
						int value = hits.removeFirst();
						movingSum -= value;
					}
					movingAverage = (double) movingSum / hits.size();
					
				}		
				//Feed symbol to model
				predictor.addSymbol(symbol);;
				//Predict
				predictedSymbol = predictor.predict();
			}
			
			//System.out.println("Fitness: " + movingAverage);
		
			
			curLine.close();
		}
		inputFile.close();
	}

}
