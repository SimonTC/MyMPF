package stcl.fun.whackamole.players;

import java.util.ArrayList;
import java.util.Random;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane.MaximizeAction;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Network_DataCollector;
import stcl.algo.brain.nodes.ActionNode;
import stcl.algo.brain.nodes.Sensor;
import stcl.algo.brain.nodes.UnitNode;
import stcl.fun.whackamole.Model;

public class ReactiveHTM extends Player {
	
	private Network_DataCollector brain;
	private SimpleMatrix currentState, predictedState;
	private int actionPerformedBefore;
	private int score;
	
	public ReactiveHTM(int numStates, Random rand){
		brain = setupNetwork(numStates, rand);
	}

	@Override
	public void step() {
		brain.getSensors().get(0).setInput(currentState);
		brain.getSensors().get(1).setInput(actionPerformedBefore);
		brain.step(score);
		SimpleMatrix actionOutput = brain.getSensors().get(1).getFeedbackOutput();
		actionPerformedBefore = (int) Math.round(actionOutput.get(0));
		predictedState = brain.getSensors().get(0).getFeedbackOutput();
		
	}
	
	private int getMaxID(SimpleMatrix m){
		int maxID = -1;
		double maxValue = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < m.getNumElements(); i++){
			double d = m.get(i);
			if (d > maxValue){
				maxValue = d;
				maxID = i;
			}
		}
		return maxID;
	}

	@Override
	public void giveScore(int score) {
		this.score = score;

	}

	@Override
	public void giveInfo(Model model) {
		currentState = new SimpleMatrix(model.nextState());
		currentState.reshape(1, currentState.getNumElements());
	}
	
	private Network_DataCollector setupNetwork(int inputLenght, Random rand){
		Network_DataCollector brain = new Network_DataCollector();
		
		//Create node
		UnitNode node = new UnitNode(0, 0, 0, 1);
		
		//Create sensors
		Sensor inputSensor = new Sensor(1, 0, 0, 0);
		inputSensor.initialize(inputLenght);
		Sensor actionSensor = new Sensor(2, 0, 1, 0);
		actionSensor.initialize(1);
		
		//Create action node
		ActionNode actionNode = new ActionNode(3);
		
		//Add children
		node.addChild(inputSensor);
		inputSensor.setParent(node);
		
		actionNode.addChild(actionSensor);
		actionSensor.setParent(actionNode);
		
		//Initialize nodes
		
		node.initialize(rand, (int) Math.sqrt(inputLenght), 1, 1, 4, false, true, true);
		actionNode.initialize(rand, 1, 2, 0.1, 1);
		
		brain.addNode(actionNode);
		brain.addNode(node);
		brain.addNode(inputSensor);
		brain.addNode(actionSensor);		
		
		//actionNode.setPossibleActions(createPossibleActions());
		
		return brain;
	}
	
	private ArrayList<SimpleMatrix> createPossibleActions(){
		ArrayList<SimpleMatrix> actions = new ArrayList<SimpleMatrix>();
		for (int i = 0; i < 2; i++){
			double[][] d = {{i}};
			SimpleMatrix m = new SimpleMatrix(d);
			actions.add(m);
		}
		return actions;
	}

	@Override
	public int getAction() {
		return actionPerformedBefore;
	}

	@Override
	public SimpleMatrix getPrediction() {
		return predictedState;
	}

	@Override
	public void endRound() {
		brain.newEpisode();
		
	}

	@Override
	public Network_DataCollector getBrain() {
		return brain;
	}
	
	

}
