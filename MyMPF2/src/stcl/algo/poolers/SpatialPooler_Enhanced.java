package stcl.algo.poolers;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class SpatialPooler_Enhanced extends SpatialPooler {
	


	private SimpleMatrix biasMatrix;

	public SpatialPooler_Enhanced(Random rand, int inputLength, int mapSize,
			double initialLearningRate, double stddev,
			double activationCodingFactor, int maxIterations) {
		super(rand, inputLength, mapSize, initialLearningRate, stddev,
				activationCodingFactor, maxIterations);
	}
	
	
	@Override
	public SimpleMatrix feedForward(SimpleMatrix feedForwardInputVector){
		
		//Classify input
		SimpleMatrix spatialFFOutputMatrix = super.feedForward(feedForwardInputVector);

		//Bias output by the prediction from t-1
		SimpleMatrix spatialFFOutputMatrixBiased = spatialFFOutputMatrix.elementMult(biasMatrix);	
				
		//Normalize output
		double sum = spatialFFOutputMatrixBiased.elementSum();
		spatialFFOutputMatrixBiased = spatialFFOutputMatrixBiased.scale(1/sum);

		//Orthogonalize output
		SimpleMatrix spatialFFOutputMatrixOrthogonalized =  aggressiveOrthogonalization(spatialFFOutputMatrixBiased);
		
		return spatialFFOutputMatrixOrthogonalized;
	}
	
	/**
	 * Orthgonalizes the matrix by setting all values to zero except for the highest value
	 * Only works with matrices containing non-negative values
	 * @param m
	 * @return
	 */
	private SimpleMatrix aggressiveOrthogonalization(SimpleMatrix m){
		int maxID = -1;
		int id = 0;
		double max = Double.NEGATIVE_INFINITY;
		double value = 0;
		
		for (double d : m.getMatrix().data){
			value = d;
			if (d > max){
				max = d;
				maxID = id;
			}
			id++;
		}
		
		SimpleMatrix orthogonalized = new SimpleMatrix(m.numRows(), m.numCols());
		orthogonalized.set(maxID, 1);
		return orthogonalized;
	
	}

}
