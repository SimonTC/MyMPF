package stcl.algo.brain.nodes;

import java.util.ArrayList;
import java.util.HashMap;

import org.ejml.simple.SimpleMatrix;

public abstract class Node {
	protected int id;
	protected int feedforwardOutputVectorLength; //Has to be set by classes implementing Node
	protected int feedforwardInputLength;
	protected Node parent;
	protected ArrayList<Node> children;
	protected SimpleMatrix feedforwardOutput;
	protected SimpleMatrix feedbackOutput;
	public enum NodeType {SENSOR, ACTION, UNIT};
	
	private int layer;
	protected boolean needHelp;
	protected NodeType type;
	
	public Node(int id, int layer) {
		this(id, layer, null);		
	}
	
	public Node(int id, int layer, Node parent) {
		this(id, layer, parent, new ArrayList<Node>());
	}
	
	public Node(int id, int layer,  Node parent, ArrayList<Node> children) {
		this.id = id;
		this.parent = parent;
		this.children = children;
		this.layer = layer;
	}
	
	public void setLayer(int layer){
		this.layer = layer;
	}
	
	public int getLayer(){
		return layer;
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
		String s = id + " " + type.ordinal() + " " + layer;
		return s;
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

}
