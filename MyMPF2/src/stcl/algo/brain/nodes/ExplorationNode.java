package stcl.algo.brain.nodes;

import java.util.ArrayList;
import java.util.Random;

public class ExplorationNode extends UnitNode {

	public ExplorationNode(int id) {
		super(id);
	}

	public ExplorationNode(int id, Node parent) {
		super(id, parent);
	}

	public ExplorationNode(int id, Node parent, ArrayList<Node> children) {
		super(id, parent, children);
	}
	
	public void setExplorationChance(double explorationChance){
		this.getUnit().getDecider().setExplorationChance(explorationChance);
	}

}
