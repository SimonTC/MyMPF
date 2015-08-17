package stcl.algo.brain.nodes;

import java.util.Random;

import stcl.algo.brain.nodes.Node.NodeType;

/**
 * Use this class to create Nodes from a String
 * @author Simon
 *
 */
public class NodeFactory {

	public Node buildNode(String s, Random rand, boolean randomInitialization){
		String[] data = s.split(" ");
		int typeID = Integer.parseInt(data[1]);
		NodeType type = NodeType.values()[typeID];
		
		Node n = null;
		if (!randomInitialization){
			switch(type){
			case SENSOR: n = new Sensor(s); break;
			case UNIT: n = new UnitNode(s);break;
			default: n = null; break;		
			}
		} else {
			switch(type){
			case SENSOR: n = new Sensor(s); break;
			case UNIT: n = new UnitNode(s, rand);break;
			default: n = null; break;		
			}
		}
		return n;
	}
}

