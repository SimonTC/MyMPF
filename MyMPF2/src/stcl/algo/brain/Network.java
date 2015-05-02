package stcl.algo.brain;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;
import java.util.TreeMap;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.nodes.ActionNode;
import stcl.algo.brain.nodes.Node;
import stcl.algo.brain.nodes.NodeFactory;
import stcl.algo.brain.nodes.Sensor;
import stcl.algo.brain.nodes.UnitNode;


public class Network implements Serializable{
	private static final long serialVersionUID = 1L;
	private ArrayList<Sensor> sensorLayer;
	private ArrayList<ArrayList<UnitNode>> unitLayers;
	private ArrayList<UnitNode> unitNodes;
	private ActionNode actionNode;
	private ArrayList<Node> nodes;
	
	public Network() {
		sensorLayer = new ArrayList<Sensor>();
		unitLayers = new ArrayList<ArrayList<UnitNode>>();
		unitNodes = new ArrayList<UnitNode>();
		nodes = new ArrayList<Node>();
	}
	
	/**
	 * Use this constructor to create a NEtwork based on the string created by the toString() method
	 * @param networkFileName
	 * @param rand
	 * @throws FileNotFoundException 
	 */
	public Network(String networkFileName, Random rand) throws FileNotFoundException {
		sensorLayer = new ArrayList<Sensor>();
		unitLayers = new ArrayList<ArrayList<UnitNode>>();
		unitNodes = new ArrayList<UnitNode>();
		nodes = new ArrayList<Node>();
		
		BufferedReader reader = new BufferedReader( new FileReader (networkFileName));
		String line = null;
		
		try {
			//Go to first node
			while( ( line = reader.readLine() ) != null) {
				if (line.equalsIgnoreCase("Nodes")) break;
			}
			//Create nodes
			NodeFactory factory = new NodeFactory();
			while( ( line = reader.readLine() ) != null) {
				if (line.equalsIgnoreCase("Connections")) break;
				Node n = factory.buildNode(line, rand);
				this.addNode(n);
			}
			//Connect nodes
			while( ( line = reader.readLine() ) != null ) {
				if(line.equalsIgnoreCase("Voter influence")) break;
				String[] arr = line.split(" --> ");
				int childID = Integer.parseInt(arr[0]);
				int parentID = Integer.parseInt(arr[1]);
				Node child = null;
				Node parent = null;
				int nodeCounter = 0;
				while(child == null || parent == null){
					Node n = nodes.get(nodeCounter);
					if (n.getID() == childID) child = n;
					if (n.getID() == parentID) parent = n;
					nodeCounter++;
				}
				
				child.setParent(parent);
				parent.addChild(child);
			}
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/**
	 * Add the given node to the network.
	 * Node will be placed according to its type
	 * @param node
	 */
	public void addNode(Node node){
		nodes.add(node);
		switch(node.getType()){
		case SENSOR: this.addSensor((Sensor) node); break;
		case UNIT: this.addUnitNode((UnitNode) node); break;			
		case ACTION: this.setActionNode((ActionNode) node); break;
		}
	}
	
	private void addSensor(Sensor sensor){
		sensorLayer.add(sensor);
	}
	
	private void addUnitNode(UnitNode node){
		int layer = node.getCoordinates()[2];
		while (unitLayers.size() <= layer) unitLayers.add(new ArrayList<UnitNode>());
		unitLayers.get(layer).add(node);
		unitNodes.add(node);
		if (actionNode != null){
			actionNode.addVoter(node);
		}
	}
	
	private void setActionNode(ActionNode node){
		this.actionNode = node;
		for (UnitNode n : unitNodes){
			actionNode.addVoter(n);
		}
	}
	
	public ActionNode getActionNode(){
		return actionNode;
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
		
		for (int layerID = unitLayers.size()-1; layerID >= 0; layerID--){
			ArrayList<UnitNode> layer = unitLayers.get(layerID);
			for (UnitNode n : layer){
				n.feedback();
			}			
		}	
		
		//Decide on what action to do
		if (actionNode != null) actionNode.feedback();
		for (Sensor s : sensorLayer) s.feedback();
		
		for (UnitNode n : unitNodes) n.resetActivityOfUnit();
		
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
	
	public String toString(){
		StringBuffer buffer = new StringBuffer();
		ArrayList<int[]> connections = new ArrayList<int[]>();
		
		buffer.append("Nodes\n");
		for (Node n :  nodes){
			buffer.append(n.toString());
			buffer.append("\n");
			int[] connection = new int[2];
			if (n.getParent() != null){
				connection[0] = n.getID();
				connection[1] = n.getParent().getID();
				connections.add(connection);
			}
		}
		
		buffer.append("Connections\n");
		for (int[] conn : connections){
			buffer.append(conn[0] + " --> " + conn[1] + "\n");
		}

		buffer.append("Voter influence\n");
		TreeMap<Integer, Double> influenceMap = actionNode.getInfluenceMap();
		for (Integer key : influenceMap.keySet()){
			double influence = influenceMap.get(key);
			buffer.append(key + " : " + influence + "\n");
		}
		
		return buffer.toString();
		
	}
	
	public void newEpisode(){
		for (UnitNode n : unitNodes) n.newEpisode();
	}
	


}
