package stcl.algo.brain;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.nodes.ActionNode;
import stcl.algo.brain.nodes.Sensor;
import stcl.algo.brain.nodes.UnitNode;


public class Network {
	private ArrayList<Sensor> sensorLayer;
	private ArrayList<ArrayList<UnitNode>> unitLayers;
	private ArrayList<UnitNode> unitNodes;
	private ActionNode actionNode;
	
	public Network() {
		sensorLayer = new ArrayList<Sensor>();
		unitLayers = new ArrayList<ArrayList<UnitNode>>();
		unitNodes = new ArrayList<UnitNode>();
	}
	
	public void addSensor(Sensor sensor){
		sensorLayer.add(sensor);
	}
	
	public void addUnitNode(UnitNode node, int layer){
		while (unitLayers.size() <= layer) unitLayers.add(new ArrayList<UnitNode>());
		unitLayers.get(layer).add(node);
		unitNodes.add(node);
		if (actionNode != null){
			actionNode.addVoter(node);
		}
	}
	
	public void setActionNode(ActionNode node){
		this.actionNode = node;
		for (UnitNode n : unitNodes){
			actionNode.addVoter(n);
		}
	}
	
	public ArrayList<Sensor> getSensors(){
		return sensorLayer;
	}
	
	public void step(double reward){
		feedForward(reward);
		
		feedback();
	}
	
	protected void feedForward(double reward){
		for (Sensor s : sensorLayer) s.feedforward();
		int actionPerformed = 0;
		if (actionNode != null) {
			actionNode.feedforward(reward, -1);		
		 	actionPerformed = actionNode.getCurrentAction();
		}
		
		for (ArrayList<UnitNode> layer : unitLayers){
			for (UnitNode n : layer){
				n.feedforward(reward, actionPerformed);
			}
		}
	}
	
	protected void feedback(){
		
		//Decide on what action to do
		if (actionNode != null) actionNode.feedback();
		
		for (int layerID = unitLayers.size()-1; layerID >= 0; layerID--){
			ArrayList<UnitNode> layer = unitLayers.get(layerID);
			for (UnitNode n : layer){
				n.feedback();
			}			
		}		
		for (Sensor s : sensorLayer) s.feedback();
	}
	
	
	public void setLearning(boolean learning){
		for (UnitNode n : unitNodes) n.getUnit().setLearning(learning);
	}
	
	public void flush(){
		for (UnitNode n : unitNodes) n.getUnit().flush();
	}
	
	public void setEntropyThresholdFrozen(boolean entropyThresholdFrozen) {
		for (UnitNode n : unitNodes) n.getUnit().setEntropyThresholdFrozen(entropyThresholdFrozen);
	}
	
	public void setBiasBeforePrediction(boolean biasBeforePrediction) {
		for (UnitNode n : unitNodes) n.getUnit().setBiasBeforePrediction(biasBeforePrediction);
	}
	
	public void setUseBiasedInputToSequencer(boolean flag) {
		for (UnitNode n : unitNodes) n.getUnit().setUseBiasedInputInSequencer(flag);
	}
	
	public void setUsePrediction(boolean flag) {
		for (UnitNode n : unitNodes) n.getUnit().setUsePrediction(flag);
	}
	
	public int getNumUnitNodes(){
		return unitNodes.size();
	}
	
	public ArrayList<UnitNode> getUnitNodes(){
		return unitNodes;
	}

}
