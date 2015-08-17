package stcl.fun.rps;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Network_DataCollector;
import stcl.fun.rps.rewardfunctions.RewardFunction;
import stcl.graphics.MPFGUI;



public class RPS {
	
	protected int[][] sequences;
	protected Random rand;
	protected int numExperimentsPerSequence;
	protected int trainingIterations;
	protected int evaluationIterations;
	protected double[][] sequenceScores;
	
	private MPFGUI gui;
	
	protected SequenceRunner runner;
	
	
	public RPS(SimpleMatrix[] possibleInputs, 
			int[][] sequences,
			RewardFunction[] rewardFunctions, 
			int numExperimentsPerSequence,
			int trainingIterations,
			int evaluationIterations,
			long randSeed,
			double noiseMagnitude,
			MPFGUI gui){
		rand = new Random(randSeed);
		runner = new SequenceRunner(null, possibleInputs, rewardFunctions, rand, noiseMagnitude);
		this.numExperimentsPerSequence = numExperimentsPerSequence;
		this.trainingIterations = trainingIterations;
		this.evaluationIterations = evaluationIterations;
		sequenceScores = new double[sequences.length][2];
		this.sequences = sequences;
		this.gui = gui;

	}
	
	public double[] run(Network_DataCollector brain, double explorationChance) {
		double totalFitness = 0;
		double totalPrediction = 0;
		
		for (int sequenceID = 0; sequenceID < sequences.length; sequenceID++){
			//System.out.println("Starting on sequence " + sequenceID);
			double sequenceFitness = 0;
			double sequencePrediction = 0;
			int[] curSequence = sequences[sequenceID];
			runner.setSequence(curSequence);
			
			
			for (int sequenceIteration = 0; sequenceIteration < numExperimentsPerSequence; sequenceIteration++){
				String name = sequenceID + " test " + sequenceIteration;
				
				
				//System.out.println("Starting on iteration " + sequenceIteration);
				runner.reset(false);
				brain.reinitialize();
				
				//Let it train
				brain.setUsePrediction(true);
				runExperiment(trainingIterations, brain, runner, null, name);
				
				//Evaluate
				brain.setLearning(false);
				brain.newEpisode();
				runner.reset(false);
				double[] scores = runExperiment(evaluationIterations, brain, runner, gui, name);
				double fitness = scores[1];
				double prediction = scores[0];
				sequenceFitness += fitness;
				sequencePrediction += prediction;
				
				//brain.getUnitNodes().get(1).getUnit().getDecider().printQMatrix();

				
			}
			double avgSequenceFitness = (sequenceFitness / (double)numExperimentsPerSequence);
			double avgSequencePrediction = (sequencePrediction / (double)numExperimentsPerSequence);
			totalFitness += avgSequenceFitness;
			totalPrediction += avgSequencePrediction;
			sequenceScores[sequenceID][0] = avgSequencePrediction;
			sequenceScores[sequenceID][1] = avgSequenceFitness;
		}
		double avgFitness = totalFitness / (double)sequences.length;
		double avgPrediction = totalPrediction / (double)sequences.length;
		double[] result = {avgPrediction, avgFitness};
		
		return result;
	}
	
	public double[] run(Network_DataCollector brain) {
		return this.run(brain,0);
		
	}
	
	/**
	 * Evaluates the activator on the given number of sequences.
	 * Remember to reset the runner and set the sequence before running this method
	 * @param numSequences the number times the sequence is repeated
	 * @param activator
	 * @return the score given as [avgPredictionSuccess, avgFitness]
	 */
	protected double[] runExperiment(int numSequences, Network_DataCollector activator, SequenceRunner runner, MPFGUI gui, String name){
		double totalPrediction = 0;
		double totalFitness = 0;
		for(int i = 0; i < numSequences; i++){
			activator.newEpisode();
			if (gui != null){
				gui.setSequenceName(name + " iteration " + i);
			} else {
				//We are training and should adjust the exploration rate
				//activator.getActionNode().setExplorationChance(1 - ((double) i / numSequences));
			}
			double[] result = runner.runSequence(activator, gui,(1 - ((double) i / numSequences)));
			totalPrediction += result[0];
			totalFitness += result[1];
		}
		
		double avgPrediction = totalPrediction / (double) numSequences;
		double avgFitness = totalFitness / (double) numSequences;
		
		double[] result = {avgPrediction, avgFitness};
		return result;
	}
	
	public double[][] getSequenceScores(){
		return sequenceScores;
	}

}
