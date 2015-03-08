package stcl.algo.predictors;

import org.ejml.simple.SimpleMatrix;

public interface Predictor {
	
	public SimpleMatrix predict(SimpleMatrix inputMatrix);
	
	public void flush();
	
	public SimpleMatrix getConditionalPredictionMatrix();
	
	public void printModel();
	
	public void setLearning(boolean learning);
	
	public void setLEarningRate(double learningRate);

}
