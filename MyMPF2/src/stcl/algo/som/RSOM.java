package stcl.algo.som;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class RSOM extends SomBasics {
	
	private SomMap leakyDifferencesMap;
	private SomMap oldLeakyDifferencesMap; 
	private double leakyCoefficient;
	private int columns, rows, inputLength;
	
	public RSOM(int columns, int rows, int inputLength, Random rand, double leakyCoefficient) {
		super(columns, rows, inputLength, rand);
		this.columns = columns;
		this.rows = rows;
		this.inputLength = inputLength;
		resetLeakyDifferencesMap();
		
		this.leakyCoefficient = leakyCoefficient; // TODO: Does this change during learning?
	}

	/**
	 * Resets the leaky difference maps to zero.
	 * Used when constructing the RSOM and when making ready for a new set of observations that are independent of the earlier observations
	 */
	public void resetLeakyDifferencesMap(){
		leakyDifferencesMap = new SomMap(columns, rows, inputLength);
		oldLeakyDifferencesMap = new SomMap(columns, rows, inputLength);
	}
	
	
	public SomNode step(SimpleMatrix inputVector, double learningRate, double neighborhoodRadius){
		if (learning){
			//Update leaky differences
			updateLeakyDifferences(inputVector, leakyCoefficient);
		}
		
		//Find BMU
		SomNode bmu = getBMU();		
		
		if (learning){
			//Update weight matrix
			updateWeightMatrix(bmu, inputVector, learningRate, neighborhoodRadius);
			//Save the leakyDifferences map
			oldLeakyDifferencesMap = leakyDifferencesMap;
		}
	
		return bmu;
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
				SomNode leakyDifferenceNode = oldLeakyDifferencesMap.get(col, row);
				SomNode weightNode = weightMap.get(col, row);
				
				//Calculate squared difference between input vector and weight vector
				SimpleMatrix weightDiff = weightNode.getVector().minus(inputVector);
				//weightDiff = weightDiff.elementPower(2);  //TODO: Decide wether to square or not. What do they do in th eoriginal code?
				weightDiff = weightDiff.scale(leakyCoefficient);
				
				SimpleMatrix leakyDifferenceVector = leakyDifferenceNode.getVector();
				leakyDifferenceVector = leakyDifferenceVector.scale(1-leakyCoefficient);
								
				SimpleMatrix sum = leakyDifferenceVector.plus(weightDiff);
				
				leakyDifferenceNode.setVector(sum);
				leakyDifferencesMap.set(col, row, leakyDifferenceNode);
			}
		}
	}
	
	@Override
	public SomNode weightAdjustment(SomNode n, SomNode bmu,
			SimpleMatrix inputVector, double neighborhoodRadius,
			double learningRate) {
		
		double squaredDistance = n.distanceTo(bmu);
		double squaredRadius = neighborhoodRadius * neighborhoodRadius;
		if (squaredDistance <= squaredRadius){ 
			double learningEffect = learningEffect(squaredDistance, squaredRadius);
			SimpleMatrix vector = n.getVector();
			SomNode oldLeakyDifferenceNode = oldLeakyDifferencesMap.get(n.getCol(), n.getRow());
			SimpleMatrix delta = oldLeakyDifferenceNode.getVector().scale(learningRate * learningEffect);
			vector = vector.plus(delta);
			n.setVector(vector);
		}
		return n;
		
	}
	
	/**
	 * Returns the best matching unit. The bmi is the unit with the lowest sum of values of its leaky difference vector.
	 * The error matrix i also updated in this method.
	 * @return
	 */
	@Override
	public SomNode getBMU(){
		double min = Double.POSITIVE_INFINITY;
		SomNode[] nodes = leakyDifferencesMap.getNodes();
		SomNode bmu = null;
		for (SomNode n : nodes){
			double value = n.getVector().elementSum();
			errorMatrix.set(n.getRow(), n.getCol(), value);
			if (value < min) {
				min = value;
				bmu = n;
			}
		}
		return bmu;
	}
	
	public SimpleMatrix getErrorMatrix(){
		return errorMatrix;
	}
	
	public SomMap getLeakyDifferencesMap(){
		return leakyDifferencesMap;
	}


	@Override
	public SomNode getBMU(SimpleMatrix inputVector) throws UnsupportedOperationException{
		throw new UnsupportedOperationException();
	}
	
	

	
	
}
