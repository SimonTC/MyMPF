package stcl.algo.brain.nodes;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

import org.ejml.simple.SimpleMatrix;

public abstract class Node implements Serializable {
	private static final long serialVersionUID = 1L;
	protected int id;
	protected int feedforwardOutputVectorLength; //Has to be set by classes implementing Node
	protected int feedforwardInputLength;
	protected Node parent;
	protected ArrayList<Node> children;
	protected SimpleMatrix feedforwardOutput;
	protected SimpleMatrix feedbackOutput;
	public enum NodeType {SENSOR, ACTION, UNIT};
	
	private int z, x, y;
	protected boolean needHelp;
	protected NodeType type;
	
	public Node(int id, int x, int y, int z) {
		this.id = id;
		this.z = z;
		this.x = x;
		this.y = y;
		children = new ArrayList<Node>();
	}
	
	/**
	 * Create new node from string created by the toString() or the toInitializationString() method
	 * @param s
	 */
	public Node(String s){
		String[] data = s.split(" ");
		id = Integer.parseInt(data[0]);
		int typeID = Integer.parseInt(data[1]);
		type = NodeType.values()[typeID];
		x = Integer.parseInt(data[2]);
		y = Integer.parseInt(data[3]);
		z = Integer.parseInt(data[4]);
		feedforwardInputLength = 0; //Don't read from file as this number is increased when children are added
		children = new ArrayList<Node>();
	}
	
	/**
	 * Return coordinate triple containing x, y, z coordinate
	 * @return
	 */
	public int[] getCoordinates(){
		int[] coor = {x,y,z};
		return coor;
	}
	
	public void setParent(Node parent){
		this.parent = parent;
	}
	
	public Node getParent(){
		return parent;
	}
	
	public void addChild(Node child){
		children.add(child);
		feedforwardInputLength += child.getFeedforwardOutputVectorLength();
	}
	
	public ArrayList<Node> getChildren(){
		return children;
	}
	
	public int getID(){
		return id;
	}
	
	public SimpleMatrix getFeedforwardOutput(){
		return feedforwardOutput;
	}
	
	/**
	 * Return the whole feedback output vector produced by this node
	 * @return
	 */
	public SimpleMatrix getFeedbackOutput(){
		return feedbackOutput;
	}
	
	/**
	 * Returns the part of the feedback output vector that belongs to the given child.
	 * @param childID
	 * @param vectorLength length of the vector that should be returned to the caller
	 * @return
	 */
	public SimpleMatrix getFeedbackOutputForChild(int childID){
		int startPosition= 0;
		Node child = null;
		for (Node n : children){
			child = n;
			if (n.getID() == childID) break;
			startPosition += n.getFeedforwardOutputVectorLength();
		}
		int vectorLength = child.getFeedforwardOutputVectorLength();
		SimpleMatrix outputForChild = feedbackOutput.extractMatrix(0, 1, startPosition, startPosition + vectorLength);
		return outputForChild;
	}
	
	public int getFeedforwardOutputVectorLength(){
		return feedforwardOutputVectorLength;
	}
	
	public boolean needHelp(){
		return needHelp;
	}
	
	public void setNeedHelp(boolean needHelp){
		this.needHelp = needHelp;
	}
	
	public void feedforward(){
		this.feedforward(0, -1);
	}
	
	@Override
	public String toString(){
		String s = id + " " + type.ordinal() + " " + x + " " + y + " " + z;
		return s;
	}
	
	public String toInitializationString(){
		return this.toString();
	}
	
	public abstract void feedforward(double reward, int actionPerformed);
	
	public abstract void feedback();

	/* (non-Javadoc)
	 * @see java.lang.Object#hashCode()
	 */
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + id;
		return result;
	}

	/* (non-Javadoc)
	 * @see java.lang.Object#equals(java.lang.Object)
	 */
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Node other = (Node) obj;
		if (id != other.id)
			return false;
		return true;
	}
	
	public NodeType getType(){
		return type;
	}
	
	/**
	 * Resets the node back to its original state before any learning has been performed.
	 * Use if you need to run multiple trainings from an initial state
	 */
	public abstract void reinitialize();
	
	public abstract void reinitialize(String initializationString);

}
