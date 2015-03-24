package stcl.algo.brain.nodes;

import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.Normalizer;

public class ExplorationNode extends UnitNode {
	private double explorationChance = 0;

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
		this.explorationChance = explorationChance;
	}
	
	@Override
	public void feedback() {
		super.feedback();		
		if (rand.nextDouble() < explorationChance){
			SimpleMatrix fbOutput = super.getFeedbackOutput();
			fbOutput.set(0);
			fbOutput.set(rand.nextInt(fbOutput.getNumElements()), 1);
		}
	}

}
