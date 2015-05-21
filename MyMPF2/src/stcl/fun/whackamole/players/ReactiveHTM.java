package stcl.fun.whackamole.players;

import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Network_DataCollector;
import stcl.algo.brain.nodes.ActionNode;
import stcl.algo.brain.nodes.Sensor;
import stcl.algo.brain.nodes.UnitNode;
import stcl.fun.whackamole.Model;

public class ReactiveHTM extends Player {
	
	private Network_DataCollector brain;
	private SimpleMatrix currentState;
	
	public ReactiveHTM(int numStates, Random rand){
		brain = setupNetwork(numStates, rand);
	}

	@Override
	public int[] action() {
		brain.getSensors().get(0).setInput(currentState);
		
	}

	@Override
	public void giveScore(int score) {
		

	}

	@Override
	public void giveInfo(Model model) {
		// TODO Auto-generated method stub

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
		
		node.initialize(rand, (int) Math.sqrt(inputLenght), 1, 1, 4, false);
		actionNode.initialize(rand, 1, 2, 0.1, 1);
		
		brain.addNode(actionNode);
		brain.addNode(node);
		brain.addNode(inputSensor);
		brain.addNode(actionSensor);		
		
		actionNode.setPossibleActions(createPossibleActions());
		
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
	
	

}
