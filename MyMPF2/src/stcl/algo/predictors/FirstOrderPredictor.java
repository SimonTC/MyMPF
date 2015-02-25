package stcl.algo.predictors;

import org.ejml.simple.SimpleMatrix;

public class FirstOrderPredictor {

	private BinaryTransitionMatrix conditionalPredictionMatrix;
	private int predictionMatrixSize;
	private double decayFactor;
	private SimpleMatrix inputVectorBefore;

	public FirstOrderPredictor(int inputMatrixSize) {
		inputVectorBefore = new SimpleMatrix(1, inputMatrixSize * inputMatrixSize);
		predictionMatrixSize = inputMatrixSize * inputMatrixSize;
		conditionalPredictionMatrix = new BinaryTransitionMatrix(predictionMatrixSize, 100); //TODO: Move to parameter
		this.decayFactor = 0.95; //TODO: Set as parameter. Parameter taken form original predictor code
	} 
	
	/**
	 * Resets the memory of the earlier input vector.
	 * DOes not change conditional memory
	 */
	public void flush(){
		inputVectorBefore.set(0);
	}
	
	/**
	 * Predicts which spatial SOM model will be active at time t+1 given the current input at time t.
	 * To correctly be able to predict staying in the same state, the input has to be highly orthogonal. That is, one element == 1 and the rest == 0
	 * @param currentStateProbabilitiesMatrix matrix containing the probabilities that model (i,j) in the spatial som is the correct model for the input to the spatial pooler at time t.
	 * @param curLearningRate
	 * @param associate if this is true the association matrix will be updated. Should be false when learning is disabled
	 * @return matrix[I x J] containing the probability that model (i,j) in the spatial som will be active at time t + 1.
	 */
	public SimpleMatrix predict(SimpleMatrix currentStateProbabilitiesMatrix, double curLearningRate, boolean associate){
		
		//Transform input matrix to vector of size IJ
		SimpleMatrix currentStateProbabilitiesVector = new SimpleMatrix(currentStateProbabilitiesMatrix);
		currentStateProbabilitiesVector.reshape(1, currentStateProbabilitiesMatrix.numCols() * currentStateProbabilitiesMatrix.numRows());
		
		//Association
		if (associate){
			conditionalPredictionMatrix.decayProbabilityMatrix(decayFactor);
			association(currentStateProbabilitiesVector, curLearningRate);
		}
		
		//Prediction
		SimpleMatrix output = prediction(currentStateProbabilitiesVector);
		
		//Transform outputVector to matrix
		output.reshape(currentStateProbabilitiesMatrix.numRows(), currentStateProbabilitiesMatrix.numCols());		
		
		//Save input vector
		inputVectorBefore = currentStateProbabilitiesVector;

		
		//Return output
		return output;
	}
	
	/**
	 * Associates the current som activations with the som activations at time t-1
	 * @param currentStateProbabilitiesVector
	 * @param curLearningRate
	 */
	private void association(SimpleMatrix currentStateProbabilitiesVector, double curLearningRate){
		for (int h = 0; h < currentStateProbabilitiesVector.numCols(); h++){
			double delta1 = inputVectorBefore.get(h) - currentStateProbabilitiesVector.get(h);
			if (delta1 < 0) delta1 = 0;
			for (int k = 0; k <currentStateProbabilitiesVector.numCols(); k++){
				double now = currentStateProbabilitiesVector.get(k);
				double before = inputVectorBefore.get(k);
				double delta2 = now - before;
				if (delta2 < 0) delta2 = 0;
				if (k==h && now ==1 && before == 1 ){ //Have to make sure that transitions to itself are also recorded
					delta1 = 1;
					delta2 = 1;
				}
				double value = delta1 * delta2;
				conditionalPredictionMatrix.addTransition(h, k, value);
			}			
		}
		
		conditionalPredictionMatrix.updateTransitionProbabilityMatrix(curLearningRate);

	}
	
	/**
	 * Predict the som activations at time t+1 given the current input 
	 * @param currentStateProbabilitiesVector
	 * @return
	 */
	private SimpleMatrix prediction(SimpleMatrix currentStateProbabilitiesVector){
		SimpleMatrix outputVector = new SimpleMatrix(currentStateProbabilitiesVector);
		double sum = 0;
		for (int k = 0; k < outputVector.numCols(); k++){
			//Calculate P(W_k) = P(W_k | W_0) x P(W_0) +  P(W_k | W_1) x P(W_1) + ... + P(W_k | W_H) x P(W_H)
			
			//First collect all P(W_k | W_h)
			SimpleMatrix conditionalPropbVector = conditionalPredictionMatrix.extractVector(true, k);
						
			//Then multiply by P(W_h)
			SimpleMatrix probabilityVector = conditionalPropbVector.elementMult(currentStateProbabilitiesVector);
			
			//Then calculate sum and add to outputVector
			double tmp = probabilityVector.elementSum();
			outputVector.set(k, tmp);
			
			sum += tmp;
		}
		
		//Normalize output vector
		if (sum == 0){ //With no prior knowledge everything is possible
			outputVector.set(1);
			sum = outputVector.elementSum();
		}
		
		outputVector = outputVector.scale(1/sum);
			
			
		
		
		return outputVector;		
	}
	
	public SimpleMatrix getConditionalPredictionMatrix(){
		return this.conditionalPredictionMatrix.getTransitionProbabilityMatrix();
	}

}
