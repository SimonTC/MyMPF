package stcl.algo.brain;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Random;
import java.util.TreeMap;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane.MaximizeAction;

import org.ejml.simple.SimpleMatrix;
import org.w3c.dom.NodeList;

import dk.stcl.core.utils.SomConstants;
import stcl.algo.brain.nodes.ActionNode;
import stcl.algo.brain.nodes.Node;
import stcl.algo.brain.nodes.Node.NodeType;
import stcl.algo.brain.nodes.NodeFactory;
import stcl.algo.brain.nodes.Sensor;
import stcl.algo.brain.nodes.UnitNode;


public class Network implements Serializable, INetwork{
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
	 * Use this constructor to create a NEtwork based on the string created by the toString() method.
	 * The spatial and temporal pooler is initialized randomly
	 * @param networkFileName
	 * @param rand
	 * @throws FileNotFoundException 
	 */
	public Network(String networkFileName, Random rand) throws FileNotFoundException {
		BufferedReader reader = new BufferedReader( new FileReader (networkFileName));
		buildNetworkFromString(reader, rand, true);
	}
	
	/* (non-Javadoc)
	 * @see stcl.algo.brain.INetwork#initialize(java.lang.String, java.util.Random)
	 */
	@Override
	public void initialize(String initializationString, Random rand){
		this.initialize(initializationString, rand, false);
	}
	
	/* (non-Javadoc)
	 * @see stcl.algo.brain.INetwork#initialize(java.lang.String, java.util.Random, boolean)
	 */
	@Override
	public void initialize(String initializationString, Random rand, boolean fromFile){
		BufferedReader reader = null;
		try {
			if (!fromFile) {
				reader = new BufferedReader( new StringReader(initializationString));
			} else {
				reader = new BufferedReader(( new FileReader(initializationString)));				
			}
			buildNetworkFromString(reader, rand, false);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void buildNetworkFromString(BufferedReader reader, Random rand, boolean initializeRandomly){
		//Build the architecture of the network
		buildArchitecture(reader, rand, initializeRandomly);
		
		if (!initializeRandomly){
			//Set all internal parameters as they where when network was exported
			setInternalParameters(reader);
		} 
	}
	
	private void buildArchitecture(BufferedReader reader, Random rand, boolean randomInitialization){
		sensorLayer = new ArrayList<Sensor>();
		unitLayers = new ArrayList<ArrayList<UnitNode>>();
		unitNodes = new ArrayList<UnitNode>();
		nodes = new ArrayList<Node>();
		
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
				Node n = factory.buildNode(line, rand, randomInitialization);
				this.addNode(n);
			}
			//Connect nodes
			while( ( line = reader.readLine() ) != null ) {
				if(line.equalsIgnoreCase("Voter influence") || line.equalsIgnoreCase("")) break;
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

	private void setInternalParameters(BufferedReader reader){
		String line = null;
		
		try {
			//Go to start of detailed description
			while( ( line = reader.readLine() ) != null) {
				if (line.equalsIgnoreCase("----Detailed description----")) break;
			}
			
			while( ( line = reader.readLine() ) != null) {
				if (line.equalsIgnoreCase("----End network description----")) break;
				if (line.contains("Node")){
					String nodeID = line.replace("Node ", "");
					reinitializeNode(nodeID, reader);
				}
			}
			
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void reinitializeNode(String NodeID, BufferedReader reader) throws IOException{
		//Find node
		int id = Integer.parseInt(NodeID);
		Node node = null;
		for (Node n : nodes){
			if (n.getID() == id){
				node = n;
				break;
			}
		}
		
		String line = reader.readLine();
		
		switch(node.getType()){
		case SENSOR: node.reinitialize(line); break;
		case UNIT:		
		case ACTION: 
			String initializationString = line + SomConstants.LINE_SEPARATOR;
			while( ( line = reader.readLine() ) != null) {
				if (line.equals("")) break;
				initializationString += line + SomConstants.LINE_SEPARATOR;
			}
			node.reinitialize(initializationString);
		}
		
	}
	
	/* (non-Javadoc)
	 * @see stcl.algo.brain.INetwork#reinitialize()
	 */
	@Override
	public void reinitialize(){
		for (Node n : nodes) n.reinitialize();
	}
	
	/* (non-Javadoc)
	 * @see stcl.algo.brain.INetwork#addNode(stcl.algo.brain.nodes.Node)
	 */
	@Override
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
	
	/* (non-Javadoc)
	 * @see stcl.algo.brain.INetwork#getActionNode()
	 */
	@Override
	public ActionNode getActionNode(){
		return actionNode;
	}
	
	/* (non-Javadoc)
	 * @see stcl.algo.brain.INetwork#getSensors()
	 */
	@Override
	public ArrayList<Sensor> getSensors(){
		return sensorLayer;
	}
	
	/* (non-Javadoc)
	 * @see stcl.algo.brain.INetwork#step(double)
	 */
	@Override
	public void step(double reward){
		feedForward(reward);
		
		feedback();
		
		resetUnitActivity();
	}
	
	/* (non-Javadoc)
	 * @see stcl.algo.brain.INetwork#feedForward(double)
	 */
	@Override
	public void feedForward(double reward){
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
	
	/* (non-Javadoc)
	 * @see stcl.algo.brain.INetwork#feedback()
	 */
	@Override
	public void feedback(){
		
		for (int layerID = unitLayers.size()-1; layerID >= 0; layerID--){
			ArrayList<UnitNode> layer = unitLayers.get(layerID);
			for (UnitNode n : layer){
				n.feedback();
			}			
		}	
		
		//Decide on what action to do
		if (actionNode != null) actionNode.feedback();
		for (Sensor s : sensorLayer) s.feedback();		
		
	}
	
	/* (non-Javadoc)
	 * @see stcl.algo.brain.INetwork#resetUnitActivity()
	 */
	@Override
	public void resetUnitActivity(){
		for (UnitNode n : unitNodes) n.resetActivityOfUnit();
	}
	
	
	/* (non-Javadoc)
	 * @see stcl.algo.brain.INetwork#setLearning(boolean)
	 */
	@Override
	public void setLearning(boolean learning){
		for (Node n : nodes) n.setLearning(learning);
	}
	
	/* (non-Javadoc)
	 * @see stcl.algo.brain.INetwork#setEntropyThresholdFrozen(boolean)
	 */
	@Override
	public void setEntropyThresholdFrozen(boolean entropyThresholdFrozen) {
		for (UnitNode n : unitNodes) n.getUnit().setEntropyThresholdFrozen(entropyThresholdFrozen);
	}
	
	/* (non-Javadoc)
	 * @see stcl.algo.brain.INetwork#setUsePrediction(boolean)
	 */
	@Override
	public void setUsePrediction(boolean flag) {
		for (UnitNode n : unitNodes) n.getUnit().setUsePrediction(flag);
	}
	
	/* (non-Javadoc)
	 * @see stcl.algo.brain.INetwork#getNumUnitNodes()
	 */
	@Override
	public int getNumUnitNodes(){
		return unitNodes.size();
	}
	
	/* (non-Javadoc)
	 * @see stcl.algo.brain.INetwork#getUnitNodes()
	 */
	@Override
	public ArrayList<UnitNode> getUnitNodes(){
		return unitNodes;
	}
	
	/* (non-Javadoc)
	 * @see stcl.algo.brain.INetwork#toString()
	 */
	@Override
	public String toString(){
		StringBuffer buffer = new StringBuffer();
		ArrayList<int[]> connections = new ArrayList<int[]>();
		String ls = SomConstants.LINE_SEPARATOR;
		
		buffer.append("----Simple description----" + ls);
		
		buffer.append("Nodes" + ls);
		for (Node n :  nodes){
			buffer.append(n.toString());
			buffer.append(ls);
			int[] connection = new int[2];
			if (n.getParent() != null){
				connection[0] = n.getID();
				connection[1] = n.getParent().getID();
				connections.add(connection);
			}
		}
		
		buffer.append("Connections" + ls);
		for (int[] conn : connections){
			buffer.append(conn[0] + " --> " + conn[1] + ls);
		}
		
		buffer.append(ls);
		buffer.append("----Detailed description----" + ls);
		this.reinitialize(); //Need to reinitialize to make sure we get the start values and not the learned values
		for (Node n : nodes){
			buffer.append("Node " + n.getID() + ls);
			String initializationString = "";
			switch (n.getType()){
			case ACTION: 
				ActionNode an = (ActionNode) n;
				initializationString = an.toInitializationString();
				break;
			case SENSOR: 
				Sensor s = (Sensor) n;
				initializationString = s.toInitializationString();
				break;
			case UNIT:
				UnitNode un = (UnitNode) n;
				initializationString = un.toInitializationString();
				break;			
			}
			
			buffer.append(initializationString );
			buffer.append(ls);
		}
		
		buffer.append("----End network description----" + ls);

		return buffer.toString();
		
	}
	
	/* (non-Javadoc)
	 * @see stcl.algo.brain.INetwork#newEpisode()
	 */
	@Override
	public void newEpisode(){
		for (UnitNode n : unitNodes) n.newEpisode();
	}
	
	/* (non-Javadoc)
	 * @see stcl.algo.brain.INetwork#toVisualString(int, int, boolean)
	 */
	@Override
	public String toVisualString(int nodeSize, int maxWidth, boolean threeD){
		StringBuffer buffer = new StringBuffer();
		//Create list of connections
		for (Node n : nodes){
			Node parent = n.getParent();
			if (parent != null){
				if (parent.getType() != NodeType.ACTION){
					buffer.append(n.getID() + " " + parent.getID());
					buffer.append("\n");
				}
			}
		}
		
		//Create the nodes
		int[] maxCoordinates = new int[3];
		for (Node n : nodes){
			int id = n.getID();
			buffer.append("//NODECLASS " + id + " " + n.getType().name() + " Nodes\n");
			buffer.append("//NODESIZE " + id + " " + nodeSize + "\n");
		}
		
		//Unit nodes should be visualized within the same area as the sensor nodes. Otherwise the visualization is too big without extra info
		
		//Find max x and y in the sensor layer
		int maxX = 0, maxY = 0;
		for (Node n : sensorLayer){
			int[] coordinates = n.getCoordinates();
			int x = coordinates[0];
			int y = coordinates[1];
			if (x > maxX) maxX = x;
			if (y > maxY) maxY = y;
		}
		
		int maxZ = unitLayers.size();
		
		//Add node coordinates
		int stepX = maxWidth / maxX; 
		int stepY = maxWidth / maxY;
		int stepZ = maxWidth / maxZ;
		for (Node n : nodes){
			int[] coordinates = n.getCoordinates();
			int x, y, z;
			if (threeD){
				x = coordinates[0] * stepX;
				y = coordinates[1] * stepY;
				z = coordinates[2] * stepZ;
			} else {
				x = (coordinates[1] * maxX + coordinates[0]) * stepX; 
				y = coordinates[2] * stepZ;
				z = maxWidth / 2;
			}
			buffer.append("//NODECOORD " + n.getID() + " " + x + " " + y + " " + z +"\n" );			
		}
		
		buffer.append("//NODECLASSCOLOR SENSOR Nodes #ff0000\n");
		buffer.append("//NODECLASSCOLOR UNIT Nodes #0000ff\n");
		buffer.append("//NODECLASSCOLOR ACTION Nodes #33FFFF\n"); //Only added to make sure they are not here
		buffer.append("//CURRENTCLASSSET Nodes");
		return buffer.toString();
	}

}
