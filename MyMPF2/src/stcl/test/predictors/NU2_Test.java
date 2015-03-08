package stcl.test.predictors;

import java.util.ArrayList;
import java.util.Random;
import java.util.Stack;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.NU;
import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.poolers.SOM;
import stcl.algo.util.Normalizer;
import dk.stcl.core.basic.containers.SomNode;

public class NU2_Test {
	
	private Random rand = new Random(1234);
	private Stack<Level> levels;
	private int[] finalSequence;
	private NU nu;
	private double curPrediction;
	
	private double noiseFactor_Eval = 0.0;
	private double noiseFactor_Train = 0.0;
	
	private SimpleMatrix equalDistribution;

	public static void main(String[] args) {
		NU2_Test t = new NU2_Test();
		t.run();

	}
	
	public void run(){
		
	
		int maxIterations = 10;

		for (double biasFactor = 0.0; biasFactor <= 1; biasFactor = biasFactor + 0.1){
			double totalError = 0;
			for (int i = 0; i < maxIterations; i++){
				
				setupExperiment(biasFactor);
				
				runTraining(noiseFactor_Train, 20);	
				
				//nu.printModel();
				
				//som.setLearning(false);
				
				double error = runEvaluation(noiseFactor_Eval, 20, biasFactor);
				
				totalError += error;
			}
			double avgMSQE = totalError / (double) maxIterations;
			System.out.println(avgMSQE);
		}
	}
	
	private void setupExperiment(double biasFactor){
		buildSequence();
		int temporalMapSize = 4;
		int inputLength = 1;
		int spatialMapSize = 4;
		double predictionLearningRate = 0.1;
		int markovOrder = 5;
		double decayFactor = 0.3;
		
		nu = new NeoCorticalUnit(rand, inputLength, spatialMapSize, temporalMapSize, predictionLearningRate, true, markovOrder);
		
		
		equalDistribution = new SimpleMatrix(temporalMapSize, temporalMapSize);
		equalDistribution.set(1);
		equalDistribution = Normalizer.normalize(equalDistribution);
	}
	
	private void printSOMMap(SOM som){
		System.out.println("SOM weights");
		String s = "";
		for (SomNode n : som.getSomMap().getNodes()){
			s += n.getVector().get(0) + "  ";
		}
		System.out.println(s);
	}
	
	private double runEvaluation(double noise, int iterations, double biasFactor){
		
		double MSQE = 0;
		for (int iteration = 0; iteration < iterations; iteration++){
			double error = doSequence(noise, biasFactor);
			
			MSQE += error / finalSequence.length;
		}
		return MSQE / (double) iterations;
	}
	
	private double doSequence(double noise, double biasFactor){
		double totalError = 0;
		for (int i : finalSequence){
			double error = Math.pow(curPrediction - (double)i, 2);
			totalError += error;
			double d = i + (0.5 - rand.nextDouble()) * noise;
			//if( biasFactor > 0) System.out.print(d + " ");
			double[][] input = {{d}};
			
			SimpleMatrix inputVector = new SimpleMatrix(input);

			SimpleMatrix spatialOutput = nu.feedForward(inputVector);
			
			SimpleMatrix fbOut = nu.feedBackward(equalDistribution);
			
			curPrediction = fbOut.get(0);	
			//if( biasFactor > 0) System.out.println(curPrediction);
		}
		return totalError;
	}
	
	private void runTraining(double noise, int iterations){
		for (int iteration = 0; iteration < iterations; iteration++){
			for (int i : finalSequence){
				double d = i + (0.5 - rand.nextDouble()) * noise;
				double[][] input = {{d}};
				
				SimpleMatrix inputVector = new SimpleMatrix(input);

				SimpleMatrix ffOut = nu.feedForward(inputVector);
				
				SimpleMatrix fbOut = nu.feedBackward(equalDistribution);
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
