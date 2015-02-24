package stcl.algo.predictors;

import org.ejml.simple.SimpleMatrix;

public class CopyOfFirstOrderPredictor {

	private SimpleMatrix conditionalPredictionMatrix;
	private int predictionMatrixSize;
	
	private SimpleMatrix inputVectorBefore;

	public CopyOfFirstOrderPredictor(int inputMatrixSize) {
		inputVectorBefore = new SimpleMatrix(1, inputMatrixSize * inputMatrixSize);
		predictionMatrixSize = inputMatrixSize * inputMatrixSize;
		conditionalPredictionMatrix = new SimpleMatrix(predictionMatrixSize, predictionMatrixSize);
		//conditionalPredictionMatrix.set(1); //Initialize to 1. 
											//TODO: Does this make sense?
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
		
		/*
		System.out.println("Input vector before");
		inputVectorBefore.print();
		System.out.println();
		
		
		System.out.println("Input vector now");
		inputVector.print();
		System.out.println();
		*/
		
		//Association
		if (associate){
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
		if (sum == 0){ //With no prior knowledge everything is possible
			outputVector.set(1);
			sum = outputVector.elementSum();
		}
		
		outputVector = outputVector.scale(1/sum);
			
			
		
		
		return outputVector;		
	}
	
	public SimpleMatrix getConditionalPredictionMatrix(){
		return this.conditionalPredictionMatrix;
	}

}
