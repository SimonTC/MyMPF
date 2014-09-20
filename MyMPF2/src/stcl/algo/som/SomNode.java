package stcl.algo.som;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class SomNode {
	
	private SimpleMatrix valueVector;
	private int col, row; // Coordinates of the node
	
	/**
	 * Creates a node with a vector with random values between 0 and 1
	 * @param vectorSize 
	 * @param rand
	 * @param col
	 * @param row
	 */
	public SomNode(int vectorSize, Random rand, int col, int row) {
		// Create vector with the dimension values and set values between 0 and 1
		valueVector = SimpleMatrix.random(1, vectorSize, 0, 1, rand);	
		setCoordinate(col, row);
	}
	
	/**
	 * Creates a new node where its internal vector is referencing the given vector
	 * @param vector
	 * @param col
	 * @param row
	 */
	public SomNode(SimpleMatrix vector, int col, int row){
		this.valueVector = vector;
		setCoordinate(col, row);
	}
	
	/**
	 * Creates a new node with a value vector containing the given data
	 * @param vector
	 * @param col
	 * @param row
	 */
	public SomNode(double[] vectorData, int col, int row){
		//Create 1D vector
		valueVector = new SimpleMatrix(1, vectorData.length);
		
		//Add data
		for (int i = 0; i< vectorData.length; i++){
			valueVector.set(0, i, vectorData[i]);
		}
		setCoordinate(col, row);
	}
	
	
	/**
	 * Only used if node is not placed in a map
	 * @param vector
	 */
	public SomNode(SimpleMatrix vector){
		this.valueVector = vector;
	}
	
	private void setCoordinate(int col, int row){
		this.col = col;
		this.row = row;
	}
	
	/**
	 * Adjust the values of the nodes based on the difference between the valueVectors of this node and input vector
	 * @param inputVector
	 * @param learningRate
	 * @param learningEffect How effective the learning is. This is dependant on the distance to the bmu
	 */
	public void adjustValues(SimpleMatrix inputVector, double learningRate, double learningEffect){
		//Calculate difference between input and current values
		SimpleMatrix diff = inputVector.minus(valueVector);
		
		//Multiply by learning rate and learning effect
		SimpleMatrix tmp = new SimpleMatrix(diff.numRows(), diff.numCols());
		tmp.set(learningRate * learningEffect);
		diff = diff.elementMult(tmp);
		
		//Add the dist-values to the value vector
		valueVector = valueVector.plus(diff);
	}
	
	/**
	 * Returns the vector with the values for this node
	 * @return
	 */
	public SimpleMatrix getVector(){
		return valueVector;
	}
	
	/**
	 * Calculates the squared difference between the values of the two nodes.
	 * @param n
	 * @return
	 */
	public double squaredDifference(SomNode n){
		SimpleMatrix thatVector = n.getVector();
		SimpleMatrix diff = valueVector.minus(thatVector);
		diff = diff.elementPower(2);
		return diff.elementSum();
	}
	
	/**
	 * Calculates the euclidian distance between the two nodes.
	 * Based on the coordinates of the nodes
	 * @param n
	 * @return returns the squared distance
	 */
	public double distanceTo(SomNode n){
		int thatX = n.getCol();
		int thatY = n.getRow();
		int myX = col;
		int myY = row;
		
		int diffX = thatX - myX;
		diffX *= diffX;
		
		int diffY = thatY - myY;
		diffY *=diffY;
		
		return diffX + diffY;
	}
	
	public int getCol(){
		return col;
	}
	
	public int getRow(){
		return row;
	}
	
	
	

}
