package stcl.test.predictors;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Random;
import java.util.Stack;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SOM;
import stcl.algo.predictors.Predictor_VOMM;
import stcl.algo.util.Normalizer;

public class SOM_VOMM_Test {
	
	private Random rand = new Random();
	private Stack<Level> levels;
	private int[] finalSequence;
	private SOM som;
	private Predictor_VOMM predictor;
	private SimpleMatrix biasMatrix;
	private double curPrediction;
	

	public static void main(String[] args) {
		SOM_VOMM_Test t = new SOM_VOMM_Test();
		t.run();

	}
	
	public void run(){
		
		double biasFactor = 0;
		
		int maxIterations = 40;
		
		double totalError = 0;
		for (int i = 0; i < maxIterations; i++){
			buildSequence();
			predictor = new Predictor_VOMM(5, 0.1);
			som = new SOM(2, 1, rand, 0.1, 0.125, 1);
			double error = runExeriment(100, true);
			totalError += error;
			System.out.println(i + " " + error);
			System.out.println();
		}
		double avgMSQE = totalError / (double) maxIterations;
		System.out.println( "MSQE: " + avgMSQE);
		
		

	}
	
	private double runExeriment(int iterations, boolean bias){
		double error = 1;
		curPrediction = 0;
		for (int i = 1; i <= iterations; i++){
			error = runTraining(bias);
			
			System.out.println(error);
		}
		return error;
	}
	
	private double runTraining(boolean bias){
		double totalError = 0;
		for (int i : finalSequence){
			double error = Math.pow(curPrediction - (double)i, 2);
			totalError += error;
			double[] input = {i};
			som.step(input);
			SimpleMatrix spatialOutput = som.computeActivationMatrix();
			
			//Normalize
			spatialOutput = Normalizer.normalize(spatialOutput);
			
			//Bias
			SimpleMatrix biasedOutput = spatialOutput;
			if (biasMatrix!= null){
				if (bias) biasedOutput = spatialOutput.elementMult(biasMatrix);
			}
			
			biasedOutput = Normalizer.normalize(biasedOutput);
			
			//Predict
			biasMatrix = predictor.predict(biasedOutput, 0.1, true);
			int predictionID = predictor.getNextPredictedSymbol();
			curPrediction = som.getSomMap().get(predictionID).getVector().get(0);			
		}
		double MSQE = totalError / finalSequence.length;
		return MSQE;
	}
	
	private void buildSequence(){
		int numLevels = 3;
		int alphabetSize = 3;
		int minBlockLength = 3;
		int maxBlockLength = 3;
		
		
		levels = createLevels(numLevels, alphabetSize, minBlockLength, maxBlockLength);
		Level topLevel = levels.peek();
		finalSequence = topLevel.unpackBlock(0);
	}
	
	/**
	 * 
	 * @param numLevels
	 * @param alphabetSize
	 * @param minBlockLength
	 * @param maxBlockLength
	 * @return The top level
	 */
	private Stack<Level>  createLevels(int numLevels, int alphabetSize, int minBlockLength, int maxBlockLength){
		Stack<Level> levels = new Stack<SOM_VOMM_Test.Level>();
		
		Level firstLevel = new Level(alphabetSize, minBlockLength, maxBlockLength, null);
		levels.push(firstLevel);
		
		
		
		for (int i = 0; i < numLevels - 1; i++){
			Level newLevel = new Level(alphabetSize, minBlockLength, maxBlockLength, levels.peek());
			levels.push(newLevel);
		}
		return levels;
	}
	
	
	private class Level{
		private int[][] blocks;
		private Level child;
		
		public Level(int alphabetSize, int minBlockLength, int maxBlockLength, Level child) {
			blocks = createLevelBlocks(alphabetSize, minBlockLength, maxBlockLength);
			this.child = child;
		}
		
		private int[] unpackBlock(int blockID){
			int[] block = blocks[blockID];
			
			if (child == null) return block;
			ArrayList<int[]> blockList = new ArrayList<int[]>();
			int totalLength = 0;
			for (int i : block){
				int[] childBlock = child.unpackBlock(i);
				blockList.add(childBlock);
				totalLength += childBlock.length;
			}
			
			int[] unpackedBlock = new int[totalLength];
			int counter = 0;
			for (int[] childBlock : blockList){
				for (int i : childBlock){
					unpackedBlock[counter] = i;
					counter++;
				}
			}
			
			return unpackedBlock;
		}
		
		
		private int[][] createLevelBlocks(int alphabetSize, int minBlockLength, int maxBlockLength){
			int numBlocks = alphabetSize;
			int[][] blocks = new int[numBlocks][];
			
			for (int blockID = 0; blockID < numBlocks; blockID++){
				int blockLength = minBlockLength + rand.nextInt(maxBlockLength - minBlockLength + 1);
				int[] block = new int[blockLength];
				for (int i = 0; i < blockLength; i++){
					block[i] = rand.nextInt(alphabetSize);
				}
				blocks[blockID] = block;
			}
			
			return blocks;		
		}
	}
	
	

}
