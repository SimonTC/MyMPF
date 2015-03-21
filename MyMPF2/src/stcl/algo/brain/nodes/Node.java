package stcl.algo.brain.nodes;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

public abstract class Node {
	private int id;
	protected Node parent;
	protected ArrayList<Node> children;
	protected SimpleMatrix feedforwardOutput;
	protected SimpleMatrix feedbackOutput;
	protected boolean needHelp;
	
	public Node(int id) {
		this(id, null);		
	}
	
	public Node(int id, Node parent) {
		this(id, parent, new ArrayList<Node>());
	}
	
	public Node(int id, Node parent, ArrayList<Node> children) {
		this.id = id;
		this.parent = parent;
		this.children = children;
	}
	
	public void setParent(Node parent){
		this.parent = parent;
	}
	
	public Node getParent(){
		return parent;
	}
	
	public void addChild(Node child){
		children.add(child);
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
	
	public SimpleMatrix getFeedbackOutput(){
		return feedbackOutput;
	}
	
	public boolean needHelp(){
		return needHelp;
	}
	
	public abstract void feedforward();
	
	public abstract void feedback();

}
