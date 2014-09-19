package stcl.som;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class SomNode {
	
	private SimpleMatrix valueVector;
	private int col, row; //ID of the column and row where the node is
	
	public SomNode(int dimensionsSize, Random rand, int col, int row) {
		// Create vector with the dimension values and set values between 0 and 1
		valueVector = SimpleMatrix.random(1, dimensionsSize, 0, 1, rand);	
		setCoordinate(col, row);
	}
	
	public SomNode(SimpleMatrix vector, int col, int row){
		this.valueVector = vector;
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
	 * Adjust the values of the nodes based on the difference between the valueVectors
	 * @param inputVector
	 * @param learningRate
	 * @param learningEffect How effective the learning is. This is dependant on the distance to the bmu
	 */
	public void adjustValues(SimpleMatrix inputVector, double learningRate, double learningEffect){
		//Calculate distance between input and current values
		SimpleMatrix dist = inputVector.minus(valueVector);
		
		//Multiply by learning rate and neighbourhood distance
		SimpleMatrix tmp = new SimpleMatrix(dist.numRows(), dist.numCols());
		tmp.set(learningRate * learningEffect);
		dist = dist.elementMult(tmp);
		
		//Add the dist-values to the value vector
		valueVector = valueVector.plus(dist);
	}
	
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
		diff.elementPower(2);
		return diff.elementSum();
	}
	
	/**
	 * Calculates the euclidian distance between the two nodes.
	 * Based on the coordinates of the nodes
	 * @param n
	 * @return
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
