package stcl.algo.brain.nodes;

import java.util.Random;

import stcl.algo.brain.nodes.Node.NodeType;

/**
 * Use this class to create Nodes from a String
 * @author Simon
 *
 */
public class NodeFactory {

	public Node buildNode(String s, Random rand){
		String[] data = s.split(" ");
		int typeID = Integer.parseInt(data[1]);
		NodeType type = NodeType.values()[typeID];
		
		Node n = null;
		switch(type){
		case ACTION: n = new ActionNode(s, rand);break;
		case SENSOR: new Sensor(s); break;
		case UNIT: n = new UnitNode(s, rand);break;
		default: n = null; break;		
		}
		return n;
	}
}

