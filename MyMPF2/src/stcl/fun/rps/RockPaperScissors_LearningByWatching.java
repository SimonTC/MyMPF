package stcl.fun.rps;

import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Brain;
import stcl.algo.brain.Brain_DataCollector;
import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.brain.Network_DataCollector;
import stcl.algo.brain.nodes.Network;
import stcl.algo.brain.nodes.Sensor;
import stcl.algo.brain.nodes.UnitNode;

public class RockPaperScissors_LearningByWatching {

	private Random rand = new Random(1234);
	private Network_DataCollector brain;
	private SimpleMatrix rock, paper, scissors, blank;
	private SimpleMatrix[] sequence;
	private int[] labelSequence;
	private SimpleMatrix rewardMatrix;
	private int[] lblCounter;
	
	private final int ITERATIONS = 5000;
	
	public static void main(String[] args) {
		RockPaperScissors_LearningByWatching runner = new RockPaperScissors_LearningByWatching();
		String folder = "C:/Users/Simon/Documents/Experiments/RPS/Network";
		runner.run(folder);
	}
	
	public RockPaperScissors_LearningByWatching() {
		// TODO Auto-generated constructor stub
	}
	
	public void run(String dataFolder){
		setup(dataFolder);
		
		//Show
		
		//Train
		//brain.setBiasBeforePrediction(true);
		brain.setUseBiasedInputToSequencer(true);

		runExperiment(1000, true);
		brain.closeFiles();
		
	}
	
	private void setup(String dataFolder){
		createInputs();
		createRewardMatrix();
		
		int ffInputLength = rock.numCols() * rock.numRows();		
		
		//Create nodes
		//Create top node
		UnitNode topNode = new UnitNode(0);
		
		//Create node that combines input and action
		UnitNode combiner = new UnitNode(1, topNode);		
		
		//Create node that pools input
		UnitNode inputPooler = new UnitNode(2, combiner);		
		
		//Create node that pools actions
		UnitNode actionPooler = new UnitNode(3, combiner);		
		
		//Create the input sensor
		Sensor inputSensor= new Sensor(4, ffInputLength, inputPooler);		
		
		//Create action sensor
		Sensor actionSensor = new Sensor(5, 3, actionPooler);		
		
		//Initialize unit nodes
			//Input pooler
			int spatialMapSize_input = 3;
			int temporalMapSize_input = 2;
			int markovOrder_input = 3;
			boolean useTemporalPooler_input = true;
			inputPooler.initializeUnit(rand, ffInputLength, spatialMapSize_input, temporalMapSize_input, 0.1, true, markovOrder_input, !useTemporalPooler_input);
			
			//Action pooler
			int ffInputLength_action = actionSensor.getFeedforwardOutputVectorLength();
			int spatialMapSize_action = 2;
			int temporalMapSize_action = 2;
			int markovOrder_action = 2;
			boolean useTemporalPooler_action = true;
			actionPooler.initializeUnit(rand, ffInputLength_action, spatialMapSize_action, temporalMapSize_action, 0.1, true, markovOrder_action, !useTemporalPooler_action);
		
			//Combiner
			int ffInputLength_combiner = actionPooler.getFeedforwardOutputVectorLength() + inputPooler.getFeedforwardOutputVectorLength();
			int spatialMapSize_combiner = 5;
			int temporalMapSize_combiner = 3;
			int markovOrder_combiner = 2;
			boolean useTemporalPooler_combiner = true;
			combiner.initializeUnit(rand, ffInputLength_combiner, spatialMapSize_combiner, temporalMapSize_combiner, 0.1, true, markovOrder_combiner, !useTemporalPooler_combiner);
		
			//top node
			int ffInputLength_top = combiner.getFeedforwardOutputVectorLength();
			int spatialMapSize_top = 5;
			int temporalMapSize_top = 3;
			int markovOrder_top = 2;
			boolean useTemporalPooler_top = true;
			topNode.initializeUnit(rand, ffInputLength_top, spatialMapSize_top, temporalMapSize_top, 0.1, true, markovOrder_top, !useTemporalPooler_top);

		
		//Add children - Needs to be done in reverse order of creation to make sure that input length calculation is correct
		actionPooler.addChild(actionSensor);
		inputPooler.addChild(inputSensor);
		combiner.addChild(actionPooler);
		combiner.addChild(inputPooler);
		topNode.addChild(combiner);
		
		//Add nodes to brain
		brain = new Network_DataCollector();
		brain.addSensor(inputSensor);
		brain.addSensor(actionSensor);
		brain.addUnitNode(inputPooler, 0);
		brain.addUnitNode(actionPooler, 0);
		brain.addUnitNode(combiner, 1);
		brain.addUnitNode(topNode, 2);
		brain.initializeWriters(dataFolder, false);
	}
	
	private void runExperiment(int episodeLength, boolean printError){
		int numEpisodes = sequence.length + 1;
		
		int curInput = 0;
		double externalReward = 0;
		
		double[][] tmp = {{1,0,0}};
		SimpleMatrix actionNow = new SimpleMatrix(tmp); //m(t)
		SimpleMatrix actionNext = new SimpleMatrix(tmp); //m(t+1)
		//SimpleMatrix actionAfterNext = new SimpleMatrix(tmp); //m(t+2)

		SimpleMatrix prediction = blank;
		int predictedLabel = 0;
		
		for (int episode = 0; episode < numEpisodes; episode++){
			for (int i = 0; i < episodeLength; i++){
				//Update action chain
				actionNow = actionNext;
				actionNext = null;// actionAfterNext;
				//actionAfterNext = null;
				
				//Get input			
				SimpleMatrix input = new SimpleMatrix(sequence[curInput]);
				
				//Calculate prediction error
				SimpleMatrix diff = input.minus(prediction);
				double predictionError = diff.normF();
				
				//Calculate reward			
				if ( i > 3){ //To get out of wrong actions
					int actionID = -1;
					if (actionNow.get(0) > 0.1) actionID = 0; //Using > 0.1 to get around doubles not always being == 0
					if (actionNow.get(1) > 0.1 ) actionID = 1;
					if (actionNow.get(2) > 0.1 ) actionID = 2;
					
					externalReward = reward(labelSequence[curInput], actionID);
				}
				
				//Give inputs to brain
				ArrayList<Sensor> sensors = brain.getSensors();
				SimpleMatrix inputVector = new SimpleMatrix(1, input.getNumElements(), true, input.getMatrix().data);
				sensors.get(0).setInput(inputVector);
				sensors.get(1).setInput(actionNow);
				
				//Do one step
				brain.step(externalReward);
				
				//Collect output
				sensors = brain.getSensors();
				prediction = new SimpleMatrix(sensors.get(0).getFeedbackOutput());
				prediction.reshape(5, 5);
				actionNext = sensors.get(1).getFeedbackOutput();

				if (curInput < episode){
					//Brain chooses next action
					int max = maxID(actionNext);
					if (max != -1){
						actionNext.set(0);
						actionNext.set(max, 1);
					}
				} else {
					//We show it the next action
					int nextID = -1;
					if (curInput == lblCounter.length - 1){
						nextID = lblCounter[0];
					} else {
						nextID = lblCounter[curInput + 1];
					}
					actionNext.set(0);
					actionNext.set(nextID, 1);
				}
				
			
				if (printError) System.out.println(i + " Error: " + predictionError + " Reward: " + externalReward);
				
				curInput++;
				if (curInput >= sequence.length){
					curInput = 0;
				}

			}
		}
		System.out.println();
		printInformation();
		
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
			unit.getSequencer().printSequenceMemory();
			System.out.println();
			System.out.println("Prediction model, unit " + id);
			unit.printPredictionModel();
			System.out.println();
		}
		
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
		double noise = (rand.nextDouble() - 0.5) * 2 * noiseMagnitude;
		m = m.plus(noise);
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
		SimpleMatrix[] tmp = {rock, paper, paper, scissors};
		int[] lbl = {0,1,1,2};
		int[] lbl_counter = {1,2,2,0};
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
