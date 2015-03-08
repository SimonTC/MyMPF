package stcl.algo.predictors;

import org.ejml.simple.SimpleMatrix;

public interface Predictor {
	
	/**
	 * 
	 * @param inputMatrix Matrix of size [I,J] Contains the probabilities of each of the possible symbols (i,j) being active at time t
	 * @return Matrix of same size as the input matrix. Contains the probabilities of each of the possible symbols (i,j) being active at time t + 1
	 */
	public SimpleMatrix predict(SimpleMatrix inputMatrix);
	
	public void flush();
	
	public SimpleMatrix getConditionalPredictionMatrix();
	
	public void printModel();
	
	public void setLearning(boolean learning);
	
	public void setLEarningRate(double learningRate);

}
