package stcl.fun.rps;

import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.brain.Network;
import stcl.algo.brain.Network_DataCollector;
import stcl.algo.brain.nodes.ActionNode;
import stcl.algo.brain.nodes.Sensor;
import stcl.algo.brain.nodes.UnitNode;
import stcl.algo.poolers.Sequencer;

public class RockPaperScissors {

	private Random rand = new Random();
	private Network_DataCollector brain;
	private SimpleMatrix rock, paper, scissors, blank;
	private SimpleMatrix[] sequence;
	private int[] labelSequence;
	private SimpleMatrix rewardMatrix;
	private int[] lblCounter;
	
	private int learningIterations = 1000;
	private int trainingIterations = 10000;
	private int evaluationIterations = 1000;
	private Sensor inputSensor1, inputSensor2, actionSensor;
	
	ActionNode actionNode;
	
	public static void main(String[] args) {
		RockPaperScissors runner = new RockPaperScissors();
		String folder = "D:/Users/Simon/Documents/Experiments/RPS/Network";
		//runner.run(folder);
		runner.runMultipleExperiments(100);
	}
	
	public RockPaperScissors() {
		// TODO Auto-generated constructor stub
	}
	
	public void run(String dataFolder){
		setup(dataFolder, true);
		
		//Show
		/*
		brain.closeFiles();
		brain.setUsePrediction(false);
		runLearning(learningIterations);
		//printInformation();
		*/
		//Train
		//brain.setBiasBeforePrediction(true);
		//brain.setUseBiasedInputToSequencer(true);
		brain.openFiles(true);
		//actionNode.setExplorationChance(0);
		brain.setUsePrediction(true);
		runExperiment(trainingIterations, true);
		brain.closeFiles();
		printInformation();
		
	}
	
	private void runMultipleExperiments(int numExperiments){
		
		double[] totalResults = new double[2];
		
		for (int exp = 1; exp <= numExperiments; exp++){
			System.out.println("Starting experiment " + exp);
			setup("", false);
			
			//Show
			brain.setUsePrediction(false);
			runLearning(learningIterations);
			
			//Train
			brain.flush();
			brain.setUsePrediction(true);
			actionNode.setExplorationChance(0.05);
			runExperiment(trainingIterations, false);
			
			//Evaluate
			brain.flush();
			actionNode.setExplorationChance(0);
			brain.setLearning(false);
			double[] results = runExperiment(evaluationIterations, false);
			totalResults[0] += results[0];
			totalResults[1] += results[1];
			System.out.println("Results, experiment " + exp + ":");
			System.out.println("Avg prediction error: " + results[0]);
			System.out.println("Avg score: " + results[1]);
			System.out.println();
		}
		
		double avgPredictionError = totalResults[0] / (double) numExperiments;
		double avgScore = totalResults[1] / (double) numExperiments;
		
		System.out.println("Final results:");
		System.out.println("Avg prediction error: " + avgPredictionError);
		System.out.println("Avg score: " + avgScore);
		
		
		
	}
	
	private void setup(String dataFolder, boolean collectData){
		createInputs();
		createRewardMatrix();
		
		int ffInputLength = rock.numCols() * rock.numRows();		
		
		//Create nodes
		//Create top node
		//UnitNode topNode = new UnitNode(0);
		
		//Create node that combines input and action
		UnitNode combiner = new UnitNode(1, null);		
		
		//Create nodes that pools input
		UnitNode inputPooler1 = new UnitNode(2, combiner);		
		UnitNode inputPooler2 = new UnitNode(3, combiner);	
		
		//Create the input sensor 1
		inputSensor1 = new Sensor(5, 12, inputPooler1);		
		
		//Create the input sensor 2
		inputSensor2 = new Sensor(6, 13, inputPooler2);	
		
		//Create action sensor
		actionSensor = new Sensor(7, 3, null);

		//Create action node
		actionNode = new ActionNode(4, 0.05, actionSensor);
		int actionMapSize = 2;
		int numActions = actionMapSize * actionMapSize;
		actionNode.initialize(rand, 3, actionMapSize, 0.1);
		actionSensor.setParent(actionNode);
		
		//Initialize unit nodes
			//Input pooler
			int spatialMapSize_input = 3;
			int temporalMapSize_input = 3;
			int markovOrder_input = 2;
			boolean useTemporalPooler_input = true;
			inputPooler1.initializeUnit(rand, inputSensor1.getFeedforwardOutputVectorLength(), spatialMapSize_input, temporalMapSize_input, 0.1, true, markovOrder_input, !useTemporalPooler_input, numActions);
			inputPooler2.initializeUnit(rand, inputSensor2.getFeedforwardOutputVectorLength(), spatialMapSize_input, temporalMapSize_input, 0.1, true, markovOrder_input, !useTemporalPooler_input, numActions);
			
			//Combiner
			
			int ffInputLength_combiner = inputPooler1.getFeedforwardOutputVectorLength() + inputPooler2.getFeedforwardOutputVectorLength();
			int spatialMapSize_combiner = 4;
			int temporalMapSize_combiner = 3;
			int markovOrder_combiner = 3;
			boolean useTemporalPooler_combiner = true;
			combiner.initializeUnit(rand, ffInputLength_combiner, spatialMapSize_combiner, temporalMapSize_combiner, 0.1, true, markovOrder_combiner, !useTemporalPooler_combiner, numActions);
		/*
			//top node
			int ffInputLength_top = combiner.getFeedforwardOutputVectorLength();
			int spatialMapSize_top = 5;
			int temporalMapSize_top = 3;
			int markovOrder_top = 2;
			boolean useTemporalPooler_top = true;
			topNode.initializeUnit(rand, ffInputLength_top, spatialMapSize_top, temporalMapSize_top, 0.1, true, markovOrder_top, !useTemporalPooler_top, numActions);
			*/
		
		//Add children - Needs to be done in reverse order of creation to make sure that input length calculation is correct
		actionNode.addChild(actionSensor);
		inputPooler1.addChild(inputSensor1);
		inputPooler2.addChild(inputSensor2);
		combiner.addChild(inputPooler1);
		combiner.addChild(inputPooler2);
		//topNode.addChild(combiner);
		
		//Add nodes to brain
		brain = new Network_DataCollector();
		brain.addSensor(inputSensor1);
		brain.addSensor(inputSensor2);
		brain.addSensor(actionSensor);
		brain.addUnitNode(inputPooler1, 0);
		brain.addUnitNode(inputPooler2, 0);
		brain.addUnitNode(combiner, 1);
		//brain.addUnitNode(topNode, 2);
		brain.setActionNode(actionNode);
		if (collectData) brain.initializeWriters(dataFolder, false);
	}
	
	private void runLearning(int iterations){
		int[][] positiveExamples = {{0,1},{1,2},{2,0}};
		int[][] negativeExamples = {{1,0},{2,1},{0,2}};
		int[][] neutralExamples = {{0,0},{1,1},{2,2}};
		
		double externalReward = 0;
		SimpleMatrix prediction = blank;
		
		double[][] tmp = {{1,0,0}};
		SimpleMatrix actionNow = new SimpleMatrix(tmp);
		
		for (int i = 0; i < iterations; i++){
			//Decide which kind of example to show
			int exampleType = rand.nextInt(3);
			
			//Decide specific example
			int[] example = null;
			switch(exampleType){
			case 0: example = positiveExamples[rand.nextInt(3)]; break;
			case 1: example = negativeExamples[rand.nextInt(3)]; break;
			case 2: example = neutralExamples[rand.nextInt(3)]; break;
			}
			
			//Get input and corresponding action	
			int inputID = example[0];
			SimpleMatrix input = new SimpleMatrix(sequence[inputID]);
			int actionID = example[1];
			actionNow.set(0);
			actionNow.set(actionID, 1);
			
			double rewardForBeingInCurrentState = externalReward;
			
			//Calculate reward			
			externalReward = reward(inputID, actionID);
			
			//Give inputs to brain
			SimpleMatrix inputVector = new SimpleMatrix(1, input.getNumElements(), true, input.getMatrix().data);
			inputSensor1.setInput(inputVector.extractMatrix(0, 1, 0, 12));
			inputSensor2.setInput(inputVector.extractMatrix(0, 1, 12, 25));
			actionSensor.setInput(actionNow);
			
			//Do one step
			brain.step(rewardForBeingInCurrentState);
		}
	}
	
	private double[] runExperiment(int maxIterations, boolean printError){
		int curInput = 0;
		double externalReward = 0;
		
		double[][] tmp = {{1,0,0}};
		SimpleMatrix actionNextTimeStep = new SimpleMatrix(tmp); //m(t)
		//SimpleMatrix actionAfterNext = new SimpleMatrix(tmp); //m(t+2)

		SimpleMatrix prediction = blank;
		
		double totalPredictionError = 0;
		double totalGameScore = 0;
		
		for (int i = 0; i < maxIterations; i++){
			//if (i % 500 == 0) System.out.println("Iteration: " + i);
			
			//Get input			
			SimpleMatrix input = new SimpleMatrix(sequence[curInput]);
			
			//Calculate prediction error
			SimpleMatrix diff = input.minus(prediction);
			double predictionError = diff.normF();	
			totalPredictionError += predictionError;
			
			SimpleMatrix actionThisTimestep = actionNextTimeStep;
			double rewardForBeingInCurrentState = externalReward;
			
			//Calculate reward			
			if ( i > 3){ //To get out of wrong actions
				int actionID = -1;
				if (actionThisTimestep.get(0) > 0.1) actionID = 0; //Using > 0.1 to get around doubles not always being == 0
				if (actionThisTimestep.get(1) > 0.1 ) actionID = 1;
				if (actionThisTimestep.get(2) > 0.1 ) actionID = 2;
				int inputID = labelSequence[curInput];
				externalReward = reward(inputID, actionID);
			}		
			
			totalGameScore += externalReward;			
			
			//Give inputs to brain
			SimpleMatrix inputVector = new SimpleMatrix(1, input.getNumElements(), true, input.getMatrix().data);
			inputSensor1.setInput(inputVector.extractMatrix(0, 1, 0, 12));
			inputSensor2.setInput(inputVector.extractMatrix(0, 1, 12, 25));
			actionSensor.setInput(actionThisTimestep);
			
			//Do one step
			brain.step(rewardForBeingInCurrentState);
			
			//Collect output
			SimpleMatrix tmp1 = inputSensor1.getFeedbackOutput();
			SimpleMatrix tmp2 = inputSensor2.getFeedbackOutput();
			prediction = tmp1.combine(0, 12, tmp2);
			//prediction = new SimpleMatrix(inputSensor.getFeedbackOutput());
			prediction.reshape(5, 5);
			
			actionNextTimeStep = actionSensor.getFeedbackOutput();

			//Set max value of action to 1. The rest to zero
			int max = maxID(actionNextTimeStep);
			if (max != -1){
				actionNextTimeStep.set(0);
				actionNextTimeStep.set(max, 1);
			}				
				
			/*
			if (i > maxIterations - 100){
				actionNode.setExplorationChance(0);
				if (printError) System.out.println(i + " Error: " + predictionError + " Reward: " + externalReward);
			}
			*/
			
			curInput++;
			if (curInput >= sequence.length){
				curInput = 0;
			}			
		}
		
		double avgPredictionError = totalPredictionError / (double) maxIterations;
		double avgScore = totalGameScore / (double) maxIterations;
		
		double[] result = {avgPredictionError, avgScore};
		return result;
	}

	
	private int maxID(SimpleMatrix m){
		//Transform bias matrix into vector
				double[] vector = m.getMatrix().data;
				
				//Go through bias vector until value is >= random number
				double max = Double.NEGATIVE_INFINITY;
				int maxID = -1;
				int id = 0;
				
				for (double d : vector){
					if (d > max){
						max = d;
						maxID = id;
					}
					id++;
				}
				return maxID;
	}
	
	
	private void printInformation(){
		
		for (UnitNode n : brain.getUnitNodes()){
			NeoCorticalUnit unit = n.getUnit();
			int id = n.getID();
			System.out.println("Spatial groups in unit " + id);
			unit.getSpatialPooler().printModelWeigths();
			System.out.println();
			System.out.println("Temporal groups in unit " + id);
			Sequencer sequencer = unit.getSequencer();
			if (sequencer != null) sequencer.printSequenceMemory();
			System.out.println();
			System.out.println("Prediction model, unit " + id);
			unit.printPredictionModel();
			System.out.println();
			System.out.println("Correlation matrix, unit " + id);
			unit.printCorrelationMatrix();
			System.out.println();
		}
		
		System.out.println("Spatial model in the action node");
		actionNode.printSomModels();
		System.out.println();
		System.out.println("VOter influence");
		actionNode.printVoterInfluence();
		
		//System.out.println("Sequences observed by unit 1:");
		//brain.getUnitList().get(0).getSequencer().printTrie();
		
		/*
		System.out.println();
		System.out.println("Spatial groups in unit 1");
		brain.getUnitNodes().get(0).getUnit().getSpatialPooler().printModelWeigths();
		System.out.println();
		System.out.println("Temporal groups in unit 1");
		brain.getUnitList().get(0).getSequencer().printSequenceMemory();
		System.out.println();
		System.out.println("Prediction model, unit 1");
		brain.getUnitList().get(0).printPredictionModel();
		
		
		System.out.println();
		System.out.println("Spatial groups in unit 2");
		brain.getUnitList().get(1).getSpatialPooler().printModelWeigths();
		System.out.println();
		System.out.println("Temporal groups in unit 2");
		brain.getUnitList().get(1).getSequencer().printSequenceMemory();
		System.out.println("Prediction model, unit 2");
		brain.getUnitList().get(1).printPredictionModel();
		*/
	}
	
	/**
	 * 
	 * @param opponentSymbol Symbol played by opponent
	 * @param playerSymbol SYmbol played by AI
	 * @return
	 */
	private double reward(int opponentSymbol, int playerSymbol){
		double reward = rewardMatrix.get(playerSymbol, opponentSymbol);
		return reward;
	}
	
	protected SimpleMatrix addNoise(SimpleMatrix m, double noiseMagnitude){
		
		for (int i = 0; i < m.getNumElements(); i++){
			double d = m.get(i);
			double noise = (rand.nextDouble() - 0.5) * 2 * noiseMagnitude;
			d += noise;
			m.set(i, d);
		}
		return m;
	}

	private void createInputs(){
		double[][] rockData = {
				{0,0,0,0,0},
				{0,1,1,1,0},
				{0,1,1,1,0},
				{0,1,1,1,0},
				{0,0,0,0,0}
		};
		
		rock = new SimpleMatrix(rockData);
		
		double[][] paperData = {
				{1,1,1,1,1},
				{1,1,1,1,1},
				{1,1,1,1,1},
				{1,1,1,1,1},
				{1,1,1,1,1}
		};
		
		paper = new SimpleMatrix(paperData);
		
		double[][] scissorsData = {
				{0,0,0,1,0},
				{1,0,1,0,0},
				{0,1,0,0,0},
				{1,0,1,0,0},
				{0,0,0,1,0}
		};
		
		scissors = new SimpleMatrix(scissorsData);		
		
		double[][] blankData = {
				{0,0,0,0,0},
				{0,0,0,0,0},
				{0,0,0,0,0},
				{0,0,0,0,0},
				{0,0,0,0,0}
		};
		
		blank = new SimpleMatrix(blankData);
		
		/*
		SimpleMatrix[] tmp = {rock, paper, paper, scissors};
		int[] lbl = {0,1,1,2};
		int[] lbl_counter = {1,2,2,0};
		*/
		
		SimpleMatrix[] tmp = {rock, paper, paper, scissors, paper, paper, scissors, rock};
		int[] lbl = {0,1,1,2,1,1,2,0};
		int[] lbl_counter = {1,2,2,0,2,2,0,1};
		
		lblCounter = lbl_counter;
		labelSequence = lbl;
		sequence = tmp;			
	}
	
	private void createRewardMatrix(){
		double[][]data = {
				{0,-1,1},
				{1,0,-1},
				{-1,1,0}
		};
		
		rewardMatrix = new SimpleMatrix(data);
	}
}
