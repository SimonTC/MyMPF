package stcl.algo.brain;

import java.util.ArrayList;
import java.util.Random;

import stcl.algo.brain.nodes.ActionNode;
import stcl.algo.brain.nodes.Node;
import stcl.algo.brain.nodes.Sensor;
import stcl.algo.brain.nodes.UnitNode;

public interface INetwork {

	/**
	 * Initializes the network using the initialization string.
	 * The string has to be identical with the string created by the toString() method
	 * @param initializationString
	 */
	public abstract void initialize(String initializationString, Random rand);

	public abstract void initialize(String initializationString, Random rand,
			boolean fromFile);

	/**
	 * Resets the network back to its original state before any learning has been performed.
	 * Use if you need to run multiple trainings from an initial state
	 */
	public abstract void reinitialize();

	/**
	 * Add the given node to the network.
	 * Node will be placed according to its type
	 * @param node
	 */
	public abstract void addNode(Node node);

	public abstract ActionNode getActionNode();

	public abstract ArrayList<Sensor> getSensors();

	/**
	 * Takes one step through the network.
	 * Set sensors before stepping
	 * @param reward
	 */
	public abstract void step(double reward);

	public abstract void feedForward(double reward);

	public abstract void feedback();

	public abstract void resetUnitActivity();

	public abstract void setLearning(boolean learning);

	public abstract void setEntropyThresholdFrozen(
			boolean entropyThresholdFrozen);

	public abstract void setUsePrediction(boolean flag);

	public abstract int getNumUnitNodes();

	public abstract ArrayList<UnitNode> getUnitNodes();

	public abstract String toString();

	public abstract void newEpisode();

	/**
	 * Return a string that can be used to visualize the network in BioLayout Express 3D version 3.3
	 * Link: http://www.biolayout.org/
	 * @param nodeSize
	 * @return
	 */
	public abstract String toVisualString(int nodeSize, int maxWidth,
			boolean threeD);

}