package stcl.algo.predictors;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.util.Normalizer;

public class FirstOrderMM_Original implements Predictor {

	private SimpleMatrix conditionalPredictionMatrix;
	private int predictionMatrixSize;
	
	private SimpleMatrix inputVectorBefore;
	private double decay;

	public FirstOrderMM_Original(int inputMatrixSize) {
		inputVectorBefore = new SimpleMatrix(1, inputMatrixSize * inputMatrixSize);
		predictionMatrixSize = inputMatrixSize * inputMatrixSize;
		conditionalPredictionMatrix = new SimpleMatrix(predictionMatrixSize, predictionMatrixSize);
		conditionalPredictionMatrix.set(1); //Initialize to 1. 
		conditionalPredictionMatrix = Normalizer.normalize(conditionalPredictionMatrix);
		
		this.decay = 0.95; //TODO: Move to parameter. This decay is used in original paper
											
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
	 * @param inputMatrix matrix containing the probabilities that model (i,j) in the spatial som is the correct model for the input to the spatial pooler at time t.
	 * @param curLearningRate
	 * @param associate if this is true the association matrix will be updated. Should be false when learning is disabled
	 * @return matrix[I x J] containing the probability that model (i,j) in the spatial som will be active at time t + 1.
	 */
	public SimpleMatrix predict(SimpleMatrix inputMatrix, double curLearningRate, boolean associate){
		
		//Transform input matrix to vector of size IJ
		SimpleMatrix inputVector = new SimpleMatrix(inputMatrix);
		inputVector.reshape(1, inputMatrix.numCols() * inputMatrix.numRows());

		//Association
		if (associate){
			decay();
			association(inputVector, curLearningRate);
		}
		
		//Prediction
		SimpleMatrix output = prediction(inputVector);
		
		//Transform outputVector to matrix
		output.reshape(inputMatrix.numRows(), inputMatrix.numCols());		
		
		//Save input vector
		inputVectorBefore = inputVector;
		
		/*
		System.out.println("Conditional prediction matrix");
		conditionalPredictionMatrix.print();
		System.out.println();
		
		System.out.println("*************************************************************");
		System.out.println();
		*/
		
		//Return output
		return output;
	}
	
	/**
	 * Associates the current som activations with the som activations at time t-1
	 * @param inputVector
	 * @param curLearningRate
	 */
	private void association(SimpleMatrix inputVector, double curLearningRate){
		int numberOfModels = inputVector.numCols();
		for (int h = 0; h < numberOfModels; h++){
			double delta1 = inputVectorBefore.get(h) - inputVector.get(h);
			if (delta1 < 0) delta1 = 0;
			for (int k = 0; k <numberOfModels; k++){ 
				
				if (k == h) continue; //We skip if transitions are between same state
				double now = inputVector.get(k);
				double before = inputVectorBefore.get(k);
				double delta2 = now - before;
				if (delta2 < 0) delta2 = 0;
				
				double tmp = conditionalPredictionMatrix.get(k, h) + delta1 * delta2 * curLearningRate;
				conditionalPredictionMatrix.set(k, h, tmp);
			}

		}
		
		conditionalPredictionMatrix = Normalizer.normalizeColumns(conditionalPredictionMatrix);

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
		outputVector = Normalizer.normalize(outputVector);	
		
		return outputVector;		
	}
	
	public SimpleMatrix getConditionalPredictionMatrix(){
		return this.conditionalPredictionMatrix;
	}
	
	/**
	 * Decays the conditional prediction matrix with the given decay value
	 * @param decay
	 */
	private void decay(){
		conditionalPredictionMatrix.scale(decay);
		conditionalPredictionMatrix = Normalizer.normalizeColumns(conditionalPredictionMatrix);
	}

}
