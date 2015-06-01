package stcl.fun.rps;

import java.io.File;
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

	private Random rand = new Random(1234);
	private Network_DataCollector brain;
	private SimpleMatrix rock, paper, scissors, blank;
	private SimpleMatrix[] sequence;
	private SimpleMatrix[] possibleInputs;
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
		//runner.runMultipleExperiments(100);
		runner.runMultipleExperiments(100, folder, false);
	}
	
	public RockPaperScissors() {
		// TODO Auto-generated constructor stub
	}
	
	public void run(String dataFolder){
		setup(dataFolder, true, rand);
		
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
	
	private void runMultipleExperiments(int numExperiments, String dataFolder, boolean collectData){

		double[] totalResults = new double[2];
		
		for (int exp = 1; exp <= numExperiments; exp++){
			System.out.println("Starting experiment " + exp);
			String folder = "";
			if (collectData){
				folder = dataFolder + "/Exp_" + exp;
				File f = new File(folder);
				f.mkdirs();
			}
			
			setup(folder, collectData, rand); //Make sure the same brain is created each time
			
			//if (collectData) brain.openFiles(true);
			
			//Show
			
			brain.setUsePrediction(false);
			runLearning(learningIterations,rand); //Make sure the learning is always the same
			brain.newEpisode();
			
			//Train
			brain.flush();
			brain.setUsePrediction(true);
			actionNode.setExplorationChance(0.05);
			runExperiment(trainingIterations, false);
			brain.newEpisode();
			
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
			
			if (collectData) brain.closeFiles();
		}
		
		double avgPredictionError = totalResults[0] / (double) numExperiments;
		double avgScore = totalResults[1] / (double) numExperiments;
		
		System.out.println("Final results:");
		System.out.println("Avg prediction error: " + avgPredictionError);
		System.out.println("Avg score: " + avgScore);
		
		
		
	}
	
	private void setup(String dataFolder, boolean collectData, Random rand){
		createInputs();
		createRewardMatrix();
		
		int ffInputLength = rock.numCols() * rock.numRows();		
		
		//Create nodes
		//Create top node
		UnitNode topNode = new UnitNode(0,0,0,3);
		
		//Create node that combines input and action
		UnitNode combiner = new UnitNode(1, 0, 0, 2);
		
		//Create node that pools input
		UnitNode inputPooler = new UnitNode(2, 0,0,1);		
		
		//Create the input sensor 1
		inputSensor1 = new Sensor(3,0,0,0);
		inputSensor1.initialize(12);
		
		//Create the input sensor 2
		inputSensor2 = new Sensor(4,0,1,0);	
		inputSensor2.initialize(13);
		
		//Create action sensor
		actionSensor = new Sensor(5, 0,2,0);
		actionSensor.initialize(3);

		//Create action node
		actionNode = new ActionNode(6);
		int actionMapSize = 2;
		int numActions = actionMapSize * actionMapSize;
		actionNode.initialize(rand, 3, actionMapSize, 0.1, 0.05);
		actionNode.addChild(actionSensor);
		actionSensor.setParent(actionNode);
		
		//Initialize unit nodes
			//Input pooler
			int spatialMapSize_input = 3;
			int temporalMapSize_input = 3;
			int markovOrder_input = 2;
			inputPooler.initialize(rand, ffInputLength, spatialMapSize_input, temporalMapSize_input,  markovOrder_input, numActions, true, false, false);
		
			//Combiner			
			int ffInputLength_combiner = inputPooler.getFeedforwardOutputVectorLength();
			int spatialMapSize_combiner = 4;
			int temporalMapSize_combiner = 3;
			int markovOrder_combiner = 3;
			combiner.initialize(rand, ffInputLength_combiner, spatialMapSize_combiner, temporalMapSize_combiner,  markovOrder_combiner,  numActions, true, false, false);
		
			//top node
			int ffInputLength_top = combiner.getFeedforwardOutputVectorLength();
			int spatialMapSize_top = 5;
			int temporalMapSize_top = 3;
			int markovOrder_top = 2;
			topNode.initialize(rand, ffInputLength_top, spatialMapSize_top, temporalMapSize_top, markovOrder_top, numActions, true, false, false);
			
		
		//Add children - Needs to be done in reverse order of creation to make sure that input length calculation is correct
		actionNode.addChild(actionSensor);
		actionSensor.setParent(actionNode);
		inputPooler.addChild(inputSensor1);
		inputSensor1.setParent(inputPooler);
		inputPooler.addChild(inputSensor2);
		inputSensor2.setParent(inputPooler);
		combiner.addChild(inputPooler);
		inputPooler.setParent(combiner);
		topNode.addChild(combiner);
		
		//Add nodes to brain
		brain = new Network_DataCollector();
		brain.addNode(inputSensor1);
		brain.addNode(inputSensor2);
		brain.addNode(actionSensor);
		brain.addNode(inputPooler);
		brain.addNode(combiner);
		brain.addNode(topNode);
		brain.addNode(actionNode);
		if (collectData) brain.initializeWriters(dataFolder, false);

	}
	
	private void runLearning(int iterations, Random rand){
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
			SimpleMatrix input = new SimpleMatrix(possibleInputs[inputID]);
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
		SimpleMatrix input = new SimpleMatrix(blank);
		step(input, new SimpleMatrix(tmp), 0);
		
		SimpleMatrix[] output = collectOutput();
		
		SimpleMatrix prediction = output[0];
		
		SimpleMatrix actionNextTimeStep = output[1];
		
		
		double totalPredictionError = 0;
		double totalGameScore = 0;
		
		for (int i = 0; i < maxIterations; i++){
			
			//Get input			
			input = new SimpleMatrix(sequence[curInput]);
			
			//Calculate prediction error
			if (i > 0){ //Don't check for error on the first. It will always be wrong
				SimpleMatrix diff = input.minus(prediction);
				double predictionError = diff.normF();	
				if (predictionError > 0.1) totalPredictionError += 1; //TODO: Maybe change threshold of error
			}
			
			SimpleMatrix actionThisTimestep = actionNextTimeStep;
			double rewardForBeingInCurrentState = externalReward;
			
			//Calculate reward			
			if ( i >= 0){ //First action is always wrong
				externalReward = calculateReward(actionThisTimestep, curInput);
			}		
			
			totalGameScore += externalReward;			
			
			
			//Give input and step
			step(input, actionThisTimestep, rewardForBeingInCurrentState);
			
			
			//Collect output
			output = collectOutput();
			
			prediction = output[0];
			
			actionNextTimeStep = output[1];

			//Set max value of action to 1. The rest to zero
			int max = maxID(actionNextTimeStep);
			if (max != -1){
				actionNextTimeStep.set(0);
				actionNextTimeStep.set(max, 1);
			}				
				
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
	
	private double calculateReward(SimpleMatrix action, int inputLabel){
		int actionID = -1;
		double maxValue = Double.NEGATIVE_INFINITY;
		for (int j = 0; j < action.getNumElements(); j++){
			double d = action.get(j);
			if (d > maxValue){
				maxValue = d;
				actionID = j;
			}
		}
		int inputID = labelSequence[inputLabel];
		double result = reward(inputID, actionID);
		return result;
	}
	
	private SimpleMatrix[] collectOutput(){
		SimpleMatrix tmp1 = inputSensor1.getFeedbackOutput();
		SimpleMatrix tmp2 = inputSensor2.getFeedbackOutput();
		SimpleMatrix prediction = tmp1.combine(0, 12, tmp2);
		prediction.reshape(5, 5);
		
		SimpleMatrix actionNextTimeStep = actionSensor.getFeedbackOutput();
		SimpleMatrix[] result = {prediction, actionNextTimeStep};
		return result;
	}
	
	private void step(SimpleMatrix input, SimpleMatrix action, double reward){
		SimpleMatrix inputVector = new SimpleMatrix(1, input.getNumElements(), true, input.getMatrix().data);
		inputSensor1.setInput(inputVector.extractMatrix(0, 1, 0, 12));
		inputSensor2.setInput(inputVector.extractMatrix(0, 1, 12, 25));
		actionSensor.setInput(action);
		
		brain.step(reward);
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
		SimpleMatrix[] tmp2 = {rock, paper, scissors};
		int[] lbl = {0,1,1,2,1,1,2,0};
		int[] lbl_counter = {1,2,2,0,2,2,0,1};
		
		lblCounter = lbl_counter;
		labelSequence = lbl;
		sequence = tmp;		
		possibleInputs = tmp2;
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
