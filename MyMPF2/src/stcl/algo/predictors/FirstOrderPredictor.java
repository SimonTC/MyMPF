package stcl.algo.predictors;

import org.ejml.simple.SimpleMatrix;

public class FirstOrderPredictor {

	private SimpleMatrix conditionalPredictionMatrix;
	private int predictionMatrixSize;
	
	private SimpleMatrix inputVectorBefore;

	public FirstOrderPredictor(int inputMatrixSize) {
		inputVectorBefore = new SimpleMatrix(1, inputMatrixSize * inputMatrixSize);
		predictionMatrixSize = inputMatrixSize * inputMatrixSize;
		conditionalPredictionMatrix = new SimpleMatrix(predictionMatrixSize, predictionMatrixSize);
		conditionalPredictionMatrix.set(1); //Initialize to 1. 
											//TODO: Does this make sense?
	} 
	
	/**
	 * Predicts which spatial som model will be active at time t+1 given the current input at time t.
	 * @param inputMatrix matrix containing the probabilities that model (i,j) in the spatial som is the correct model for the input to the spatial pooler at time t.
	 * @param curLearningRate
	 * @return matrix[I x J] containing the probability that model (i,j) in the spatial som will be active at time t + 1.
	 */
	public SimpleMatrix predict(SimpleMatrix inputMatrix, double curLearningRate){
		
		//Transform input matrix to vector of size IJ
		SimpleMatrix inputVector = new SimpleMatrix(inputMatrix);
		inputVector.reshape(1, inputMatrix.numCols() * inputMatrix.numRows());
		
		//Association
		association(inputVector, curLearningRate);
		
		//Prediction
		SimpleMatrix output = prediction(inputVector);
		
		//Transform outputVector to matrix
		output.reshape(inputMatrix.numRows(), inputMatrix.numCols());		
		
		//Save input vector
		inputVectorBefore = inputVector;
		
		//Return output
		return output;
	}
	
	/**
	 * Associates the current som activations with the som activations at time t-1
	 * @param inputVector
	 * @param curLearningRate
	 */
	private void association(SimpleMatrix inputVector, double curLearningRate){
		for (int h = 0; h < inputVector.numCols(); h++){
			double delta1 = inputVectorBefore.get(h) - inputVector.get(h);
			if (delta1 < 0) delta1 = 0;
			double sum = 0;
			for (int k = 0; k <inputVector.numCols(); k++){
				double now = inputVector.get(k);
				double before = inputVectorBefore.get(k);
				double delta2 = now - before;
				if (delta2 < 0) delta2 = 0;
				if (k==h && now ==1 && before == 1 ){ //Have to make sure that transitions to itself are also recorded
					delta1 = 1;
					delta2 = 1;
				}
				double tmp = conditionalPredictionMatrix.get(k, h) + delta1 * delta2 * curLearningRate;
				conditionalPredictionMatrix.set(k, h, tmp);
				sum += tmp;
			}
			if (sum > 0){
				SimpleMatrix column = conditionalPredictionMatrix.extractVector(false, h);
				column = column.scale(1/sum);
				conditionalPredictionMatrix.insertIntoThis(0, h, column);
			}
		}

	}
	
	/**
	 * Predict the som activations at time t+1 given the current input 
	 * @param inputVector
	 * @return
	 */
	private SimpleMatrix prediction(SimpleMatrix inputVector){
		SimpleMatrix outputVector = new SimpleMatrix(inputVector);
		double sum = 0;
		for (int k = 0; k < outputVector.numCols(); k++){
			//Calculate P(W_k) = P(W_k | W_0) x P(W_0) +  P(W_k | W_1) x P(W_1) + ... + P(W_k | W_H) x P(W_H)
			
			//First collect all P(W_k | W_h)
			SimpleMatrix conditionalPropbVector = conditionalPredictionMatrix.extractVector(true, k);
						
			//Then multiply by P(W_h)
			SimpleMatrix probabilityVector = conditionalPropbVector.elementMult(inputVector);
			
			//Then calculate sum and add to outputVector
			double tmp = probabilityVector.elementSum();
			outputVector.set(k, tmp);
			
			sum += tmp;
		}
		
		//Normalize output vector
		if (sum > 0){
			outputVector = outputVector.scale(1/sum);
		}
		
		return outputVector;
		
		
	}

}
