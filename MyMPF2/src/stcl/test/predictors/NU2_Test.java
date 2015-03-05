package stcl.test.predictors;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Random;
import java.util.Stack;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.NeoCorticalUnit2;
import stcl.algo.poolers.SOM;
import stcl.algo.predictors.Predictor_VOMM;
import stcl.algo.util.Normalizer;

public class NU2_Test {
	
	private Random rand = new Random(1234);
	private Stack<Level> levels;
	private int[] finalSequence;
	private NeoCorticalUnit2 nu;
	private double curPrediction;
	
	private double noiseFactor_Eval = 2.0;
	private double noiseFactor_Train = 0.0;

	public static void main(String[] args) {
		NU2_Test t = new NU2_Test();
		t.run();

	}
	
	public void run(){
		
	
		int maxIterations = 10;

		for (double biasFactor = 0; biasFactor <= 1; biasFactor = biasFactor + 0.1){
			double totalError = 0;
			for (int i = 0; i < maxIterations; i++){
				buildSequence();
				nu = new NeoCorticalUnit2(rand, 1, 3, 4, 0.1, 3, 0.3, biasFactor);
				runTraining(noiseFactor_Train, 20);
				//som.setLearning(false);
				double error = runEvaluation(noiseFactor_Eval, 20, biasFactor);
				totalError += error;
			}
			double avgMSQE = totalError / (double) maxIterations;
			System.out.println(avgMSQE);
		}
	}
	
	private double runEvaluation(double noise, int iterations, double biasFactor){
		double MSQE = 0;
		for (int iteration = 0; iteration < iterations; iteration++){
			double totalError = 0;
			for (int i : finalSequence){
				double error = Math.pow(curPrediction - (double)i, 2);
				totalError += error;
				double d = i + (0.5 - rand.nextDouble()) * noise;
				double[][] input = {{d}};
				
				SimpleMatrix inputVector = new SimpleMatrix(input);

				SimpleMatrix ffOut = nu.feedForward(inputVector);
				
				SimpleMatrix fbOut = nu.feedBackward(ffOut);
				
				curPrediction = fbOut.get(0);	
			}
			MSQE += totalError / finalSequence.length;
		}
		return MSQE / (double) iterations;
	}
	
	private void runTraining(double noise, int iterations){
		for (int iteration = 0; iteration < iterations; iteration++){
			for (int i : finalSequence){
				double d = i + (0.5 - rand.nextDouble()) * noise;
				double[][] input = {{d}};
				
				SimpleMatrix inputVector = new SimpleMatrix(input);

				SimpleMatrix ffOut = nu.feedForward(inputVector);
				
				SimpleMatrix fbOut = nu.feedBackward(ffOut);
			}
		}
	}
	
	private boolean isMatrixOK(SimpleMatrix m){
		for (double d : m.getMatrix().data){
			if (Double.isNaN(d)) return false;
			if (Double.isInfinite(d)) return false;
		}
		return true;
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
		Stack<Level> levels = new Stack<NU2_Test.Level>();
		
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
