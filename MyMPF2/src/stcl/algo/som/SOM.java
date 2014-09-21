package stcl.algo.som;

import java.util.Random;
import java.util.Vector;

import org.ejml.data.MatrixIterator;
import org.ejml.simple.SimpleMatrix;

public class SOM {
	private SomMap weightMap;
	private SimpleMatrix errorMatrix;
	
	
	/**
	 * Creates a new SOM where all node vector values are initialized to a random value between 0 and 1
	 * @param columns width of the map 
	 * @param rows height of the map 
	 */
	public SOM(int columns, int rows, int inputLength, Random rand) {
		weightMap = new SomMap(columns, rows, inputLength, rand);
		errorMatrix = new SimpleMatrix(rows, columns);
	}
	
	/**
	 * Finds the BMU to the given input input vector and updates the vectors of all the nodes
	 * @param inputVector
	 * @param learningRate
	 * @param neighborhoodRadius
	 * @return
	 */
	public SomNode step (double[] inputVector, double learningRate, double neighborhoodRadius){
		//Create input vector
		double[][] d = {inputVector};
		SimpleMatrix m = new SimpleMatrix(d);
		
		SomNode bmu = step(m, learningRate, neighborhoodRadius);

		
		return bmu;
	}
	/**
	 * Finds the BMU to the given input node and updates the vectors of all the nodes
	 * @param inputNode
	 * @param learningRate
	 * @param neighborhoodRadius
	 * @return
	 */
	public SomNode step (SimpleMatrix inputVector, double learningRate, double neighborhoodRadius){
		//Find BMU
		SomNode bmu = getBMU(inputVector);
		
		//Adjust Weights
		adjustWeights(bmu, inputVector, learningRate, neighborhoodRadius);	
		
		return bmu;
	}
		
	/**
	 * Returns the node which vector is least different from the vector of the input node. This method also updates the internal error matrix
	 * @param input input as a somNode
	 * @return
	 */
	public SomNode getBMU(SimpleMatrix inputVector){
		SomNode BMU = null;
		double minDiff = Double.POSITIVE_INFINITY;
		for (SomNode n : weightMap.getNodes()){
			double diff = n.squaredDifference(inputVector);
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
	public void adjustWeights(SomNode bmu, SimpleMatrix inputVector, double learningRate, double neighborhoodRadius){
		//Calculate start and end coordinates for the weight updates
		int bmuCol = bmu.getCol();
		int bmuRow = bmu.getRow();
		int colStart = (int) (bmuCol - neighborhoodRadius);
		int rowStart = (int) (bmuRow - neighborhoodRadius );
		int colEnd = (int) (bmuCol + neighborhoodRadius);
		int rowEnd = (int) (bmuRow + neighborhoodRadius );
		
		//Make sure we don't get out of bounds errors
		if (colStart < 0) colStart = 0;
		if (rowStart < 0) rowStart = 0;
		if (colEnd > weightMap.getWidth()) colEnd = weightMap.getWidth();
		if (rowEnd > weightMap.getHeight()) rowEnd = weightMap.getHeight();
		
		//Adjust weights
		for (int col = colStart; col < colEnd; col++){
			for (int row = rowStart; row < rowEnd; row++){
				SomNode n = weightMap.get(col, row);
				weightAdjustment(n, bmu, inputVector, neighborhoodRadius, learningRate);
			}
		}
	}
	
	public void weightAdjustment(SomNode n, SomNode bmu, SimpleMatrix inputVector, double neighborhoodRadius, double learningRate ){
		double squaredDistance = n.distanceTo(bmu);
		double squaredRadius = neighborhoodRadius * neighborhoodRadius;
		if (squaredDistance <= squaredRadius){ 
			double learningEffect = learningEffect(squaredDistance, squaredRadius);
			n.adjustValues(inputVector, learningRate, learningEffect);					
		}
	}
	
	public SomNode[] getModels(){
		return weightMap.getNodes();
	}
	
	public SimpleMatrix getErrorMatrix(){
		return errorMatrix;
	}
	
	public SomNode getModel(int id){
		return weightMap.get(id);
	}
	
	public SomNode getModel(int row, int col){
		return weightMap.get(col, row);
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
	
	public void set(SomNode n, int row, int column){
		weightMap.set(column, row, n);
	}
	
	public int getWidth(){
		return weightMap.getWidth();
	}
	
	public int getHeight(){
		return weightMap.getHeight();
	}
	
	public SomMap getMap(){
		return weightMap;
	}
	
	
	
	

}
