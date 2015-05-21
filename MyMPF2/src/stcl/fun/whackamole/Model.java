package stcl.fun.whackamole;

import java.util.Random;
import java.util.Stack;

import org.ejml.simple.SimpleMatrix;

import stcl.fun.sequenceprediction.SequenceBuilder;
import stcl.fun.sequenceprediction.SequenceLevel;

public class Model {
	private int[] moleSequence, holySequence;
	private SequenceBuilder builder;
	private int currentStep;
	private int runningScore, maxPossibleScore;
	private boolean gameOver;
	
	/**
	 * 
	 * @param activeField which field is the player aiming at
	 * @param hit is the player hitting the field
	 * @return 1 for hitting a unholy mole, 0 for hitting wrong field / not hitting at all, -1 for hitting a holy mole
	 */
	public int step(int activeField, boolean hit){
		int score = 0;
		if (!gameOver){
			int correctField = moleSequence[currentStep];
			boolean shouldHit = holySequence[currentStep] == 0; //If it is not a holy mole, you should hit it
			if (shouldHit) maxPossibleScore++;
			
			if (activeField != correctField) {
				score = 0;
			} else if (shouldHit == hit){
				score = 1;
			} else {
				score = -1;	
			}
			runningScore += score;
			currentStep++;
		}
		gameOver = (currentStep >= moleSequence.length);
		return score;
	}
	
	public boolean isGameOver(){
		return gameOver;
	}
	
	public int getMaxPossibleScore(){
		return maxPossibleScore;
	}
	
	public int getRunningScore(){
		return runningScore;
	}
	
	public void start(){
		currentStep = 0;
		runningScore = 0;
		maxPossibleScore = 0;
		gameOver = false;
	}
	
	public void initialize(int worldSize, int numLevels, int minBlockLength, int maxBlockLength, double holyChance, Random rand){
		moleSequence = buildMoleSequence(rand, (int)Math.pow(worldSize, 2), numLevels, minBlockLength, maxBlockLength);
		holySequence = buildHolySequence(builder, holyChance, rand);
	}
	
	private int[] buildMoleSequence(Random rand, int alphabetSize, int numLevels, int minBlockLength, int maxBlockLength){
		builder = new SequenceBuilder();
		int[] moleSequence = builder.buildSequence(rand, numLevels, alphabetSize, minBlockLength, maxBlockLength);
		return moleSequence;
	}
	
	private int[] buildHolySequence(SequenceBuilder builder, double holyChance, Random rand){
		SequenceLevel holyTop = new SequenceLevel(builder.getTopLevel());
		SequenceLevel parent = holyTop.getChild();
		SequenceLevel child = parent.getChild();
		
		do{
			parent = child;
			child = parent.getChild();			
		} while (child != null);
		
		int[][] holyBlocks = parent.getBlocks();
		int numBlocks = holyBlocks.length;
		
		for (int blockID = 0; blockID < numBlocks; blockID++){
			int blockLength = holyBlocks[blockID].length;
			int[] block = new int[blockLength];
			for (int i = 0; i < blockLength; i++){
				block[i] = (rand.nextDouble() < holyChance)? 1 : 0;				
			}
			holyBlocks[blockID] = block;
		}
		
		int[] holySequence = holyTop.unpackBlock(0);
		return holySequence;
	}
}
