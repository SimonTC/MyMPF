package stcl.algo.predictors;

import org.ejml.simple.SimpleMatrix;

public interface Predictor {
	
	public SimpleMatrix predict(SimpleMatrix inputMatrix, double curLearningRate, boolean associate);
	
	public void flush();
	
	public SimpleMatrix getConditionalPredictionMatrix();
	
	public void printModel();

}
