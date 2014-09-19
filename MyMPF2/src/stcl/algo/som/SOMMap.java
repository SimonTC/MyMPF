package stcl.algo.som;

import java.util.Random;
import java.util.Vector;

import org.ejml.data.MatrixIterator;
import org.ejml.simple.SimpleMatrix;

public class SOMMap {
	private int rows, columns;
	private SomNode[] models; //The models in the map
	private SimpleMatrix errorMatrix;
	
	
	/**
	 * Creates a new SOM where all node vector values are initialized to a random value between 0 and 1
	 * @param columns width of the map (coulumns)
	 * @param rows height of the map (rows)
	 */
	public SOMMap(int columns, int rows, int inputLength, Random rand) {
		this.columns = columns;
		this.rows = rows;
		initializeMap(inputLength, rand);
		errorMatrix = new SimpleMatrix(rows, columns);
	}
	
	/**
	 * Fills the map with nodes where the vector values are set to random values between 0 and 1
	 * @param columns
	 * @param rows
	 * @param inputLength
	 * @param rand
	 */
	private void initializeMap(int inputLength, Random rand){
		models = new SomNode[rows * columns];
		
		for (int row = 0; row < rows; row++){
			for (int col = 0; col < columns; col++){
				SomNode n = new SomNode(inputLength, rand, col, row);
				models[coordinateToIndex(row, col)] = n;
			}
		}
	}
	
	/**
	 * Finds the BMU to the given input and updates the vectors of all the nodes
	 * @param inputVector
	 * @param learningRate
	 * @param neighborhoodRadius
	 * @return
	 */
	public SomNode step (double[] inputVector, double learningRate, double neighborhoodRadius){
		//Create input node
		SomNode inputNode = new SomNode(inputVector, -1, -1);
		
		SomNode bmu = step(inputNode, learningRate, neighborhoodRadius);

		
		return bmu;
	}
	
	public SomNode step (SomNode inputNode, double learningRate, double neighborhoodRadius){
		//Find BMU
		SomNode bmu = getBMU(inputNode);
		
		//Adjust Weights
		adjustWeights(bmu, inputNode, learningRate, neighborhoodRadius);	
		
		return bmu;
	}
	/**
	 * Returns the node which vector is least different from the input vector
	 * @param input as a array of doubles
	 * @return
	 */
	public SomNode getBMU(double[] input){ 
		SomNode inputNode = new SomNode(input, -1, -1);
		return getBMU(inputNode);
	}
	
	/**
	 * Returns the node which vector is least different from the vector of the input node
	 * @param input input as a somNode
	 * @return
	 */
	public SomNode getBMU(SomNode input){
		SomNode BMU = null;
		double minDiff = Double.POSITIVE_INFINITY;
		for (SomNode n : models){
			double diff = n.squaredDifference(input);
			if (diff < minDiff){
				minDiff = diff;
				BMU = n;
			}
			
			errorMatrix.set(n.getRow(), n.getCol(), diff);
		}
		return BMU;
	}
	
	/**
	 * Adjusts the weigths of the som
	 * @param bmu
	 * @param learningRate
	 * @param neighborhoodRadius
	 */
	public void adjustWeights(SomNode bmu,SomNode input, double learningRate, double neighborhoodRadius){
		//Calculate start and end coordinates for the weight updates
		int colStart = (int) (bmu.getCol() - neighborhoodRadius - 1);
		int rowStart = (int) (bmu.getRow() - neighborhoodRadius - 1);
		int colEnd = (int) (colStart + (neighborhoodRadius * 2) + 1);
		int rowEnd = (int) (rowStart + (neighborhoodRadius * 2) + 1);
		
		//Make sure we dont get out of bounds errors
		if (colStart < 0) colStart = 0;
		if (rowStart < 0) rowStart = 0;
		if (colEnd > columns) colEnd = columns;
		if (rowEnd > rows) rowEnd = rows;
		
		//Adjust weights
		for (int col = colStart; col < colEnd; col++){
			for (int row = rowStart; row < rowEnd; row++){
				SomNode n = models[coordinateToIndex(row, col)];
				double squaredDistance = n.distanceTo(bmu);
				double squaredRadius = neighborhoodRadius * neighborhoodRadius;
				if (squaredDistance < squaredRadius){ 
					double learningEffect = learningEffect(squaredDistance, squaredRadius);
					n.adjustValues(input.getVector(), learningRate, learningEffect);					
				}
			}
		}
	}
	
	public SomNode[] getModels(){
		return models;
	}
	
	public SimpleMatrix getErrorMatrix(){
		return errorMatrix;
	}
	
	public SomNode getModel(int id){
		return models[id];
	}
	
	public SomNode getModel(int row, int col){
		return models[coordinateToIndex(row, col)];
	}
	
	
	/**
	 * Calculates the learning effect based on distance to the learning center.
	 * The lower the distance, the higher the learning effect
	 * @param squaredDistance
	 * @param squaredRadius
	 * @return
	 */
	private double learningEffect(double squaredDistance, double squaredRadius){
		double d = Math.exp(-(squaredDistance / (2 * squaredRadius)));
		return d;
	}
	
	private int coordinateToIndex(int row, int col){
		return (row * columns + col);
	}
	
	public void set(SomNode n, int row, int column){
		models[coordinateToIndex(row, column)] = n;
	}
	
	
	
	

}
