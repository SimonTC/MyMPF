package stcl.algo.som;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class RSOM  {
	private SomMap leakyDifferencesMap;
	private SomMap oldLeakyDifferencesMap; 
	private SomMap weightMap;
	private SimpleMatrix errorMatrix;
	
	
	public void step(SimpleMatrix inputVector, double leakyCoefficient, double learningRate, double neighborhoodRadius){
		//Update leaky differences
		updateLeakyDifferences(inputVector, leakyCoefficient);
		
		//Find BMU
		SomNode bmu = findBMU();		
		
		//Update weight matrix
		updateWeightMatrix(bmu, learningRate, neighborhoodRadius);
	
	}

	/**
	 * Updates the vector values of the nodes in the leaky differences map
	 * @param inputVector
	 * @param leakyCoefficient
	 */
	private void updateLeakyDifferences(SimpleMatrix inputVector, double leakyCoefficient){
		//Convert input vector to node to make it easier to work with
		
		for (int row = 0; row < leakyDifferencesMap.getHeight(); row++){
			for (int col = 0; col < leakyDifferencesMap.getWidth(); col++){
				SomNode leakyDifferenceNode = leakyDifferencesMap.get(col, row);
				//Calculate squared difference between input vector and weight vector
				SimpleMatrix weightDiff = weightMap.get(col, row).getVector().minus(inputVector);
				weightDiff = weightDiff.elementPower(2);
				weightDiff = weightDiff.scale(leakyCoefficient);
				
				SimpleMatrix leakyDifferenceVector = leakyDifferenceNode.getVector();
				leakyDifferenceVector = leakyDifferenceVector.scale(1-leakyCoefficient);
				leakyDifferenceVector = leakyDifferenceVector.plus(weightDiff);
				
				leakyDifferenceNode.setVector(leakyDifferenceVector);
			}
		}
	}
	
	private void updateWeightMatrix(SomNode bmu, double learningRate, double neighborhoodRadius){
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
				SomNode oldLeakyDifferenceNode = oldLeakyDifferencesMap.get(col, row);
				weightAdjustment(n, bmu, oldLeakyDifferenceNode, neighborhoodRadius, learningRate);
			}
		}
	}
	
	private void weightAdjustment(SomNode n, SomNode bmu, SomNode oldLeakyDifferenceNode, double neighborhoodRadius, double learningRate ){
		double squaredDistance = n.distanceTo(bmu);
		double squaredRadius = neighborhoodRadius * neighborhoodRadius;
		if (squaredDistance <= squaredRadius){ 
			double learningEffect = learningEffect(squaredDistance, squaredRadius);
			SimpleMatrix vector = n.getVector();
			SimpleMatrix delta = oldLeakyDifferenceNode.getVector().scale(learningRate * learningEffect);
			vector = vector.plus(delta);
			n.setVector(vector);
		}
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
	
	/**
	 * Returns the best matching unit. The bmi is the unit with the lowest sum of values of its leaky difference vector.
	 * The error matrix i also updated in this method.
	 * @return
	 */
	private SomNode findBMU(){
		double max = Double.POSITIVE_INFINITY;
		SomNode[] nodes = leakyDifferencesMap.getNodes();
		SomNode bmu = null;
		for (SomNode n : nodes){
			double value = n.getVector().elementSum();
			errorMatrix.set(n.getRow(), n.getCol(), value);
			if (value > max) {
				max = value;
				bmu = n;
			}
		}
		return bmu;
	}
	
	public SimpleMatrix getErrorMatrix(){
		return errorMatrix;
	}
	
	
	
}
