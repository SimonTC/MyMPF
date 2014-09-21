package stcl.algo.som;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class RSOM extends SomBasics {
	
	private SomMap leakyDifferencesMap;
	private SomMap oldLeakyDifferencesMap; 
	private double leakyCoefficient;
	
	public RSOM(int columns, int rows, int inputLength, Random rand, double leakyCoefficient) {
		super(columns, rows, inputLength, rand);
		leakyDifferencesMap = new SomMap(columns, rows, inputLength, rand);
		this.leakyCoefficient = leakyCoefficient; // TODO: Does this change during learning?
	}

	
	
	
	public SomNode step(SimpleMatrix inputVector, double learningRate, double neighborhoodRadius){
		//Update leaky differences
		updateLeakyDifferences(inputVector, leakyCoefficient);
		
		//Find BMU
		SomNode bmu = getBMU();		
		
		//Update weight matrix
		updateWeightMatrix(bmu, inputVector, learningRate, neighborhoodRadius);
		
		//Save the leakyDifferences map
		oldLeakyDifferencesMap = leakyDifferencesMap;
	
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
	
	@Override
	public void weightAdjustment(SomNode n, SomNode bmu,
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
		
	}
	
	/**
	 * Returns the best matching unit. The bmi is the unit with the lowest sum of values of its leaky difference vector.
	 * The error matrix i also updated in this method.
	 * @return
	 */
	@Override
	public SomNode getBMU(){
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


	@Override
	public SomNode getBMU(SimpleMatrix inputVector) throws UnsupportedOperationException{
		throw new UnsupportedOperationException();
	}

	
	
}
