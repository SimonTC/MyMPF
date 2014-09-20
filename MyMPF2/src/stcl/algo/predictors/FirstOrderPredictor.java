package stcl.algo.predictors;

import org.ejml.simple.SimpleMatrix;

public class FirstOrderPredictor {
	
	SimpleMatrix inpuMatrixBefore;
	SimpleMatrix conditionalPredictionMatrix;
	private int predictionMatrixSize;

	public FirstOrderPredictor(int inputMatrixSize) {
		inpuMatrixBefore = new SimpleMatrix(inputMatrixSize, inputMatrixSize);
		predictionMatrixSize = inputMatrixSize * inputMatrixSize;
		conditionalPredictionMatrix = new SimpleMatrix(predictionMatrixSize, predictionMatrixSize);
		// TODO Auto-generated constructor stub
	}
	
	/**
	 * 
	 * @param inputMatrix matrix containing the probabilities that model (i,j) in the spatial som is the correct model for the input to the spatial pooler.
	 * @param curLearningRate
	 * @return
	 */
	public SimpleMatrix predict(SimpleMatrix inputMatrix, double curLearningRate){
		//TODO: Check this algorithm once more. Have to be sure that it goes in right direction		
	
		for (int h = 0; h < predictionMatrixSize; h++){ //Go through rows
			double priorProbOfModelHNow =inputMatrix.get(h);
			double priorProbOfModelHBefore =inpuMatrixBefore.get(h);
			double max = Double.NEGATIVE_INFINITY;
			for (int k = 0; k < predictionMatrixSize; k++){ //Go through columns
				double priorProbOfModelKNow = inputMatrix.get(k);
				double priorProbOfModelKBefore = inpuMatrixBefore.get(k);
				double delta1 = Math.max(priorProbOfModelHBefore - priorProbOfModelHNow, 0);
				double delta2 = Math.max(priorProbOfModelKNow - priorProbOfModelKBefore, 0 );
				double conditionalProbability = conditionalPredictionMatrix.get(h, k) + curLearningRate * delta1 * delta2;
				if (conditionalProbability > max) max = conditionalProbability;
				conditionalPredictionMatrix.set(h, k, conditionalProbability);
			}
			
			//Normalize column n
			SimpleMatrix normalizedRow = conditionalPredictionMatrix.extractMatrix(h, h+1, 0, conditionalPredictionMatrix.END);
			normalizedRow = normalizedRow.scale(1/max);
			conditionalPredictionMatrix.insertIntoThis(h, 0, normalizedRow);			
		}
		
		//Calculate output
		//TODO: I think there is a mistake in the article. Has to check other prediction sources
		SimpleMatrix outPut = new SimpleMatrix(inputMatrix.numRows(), inputMatrix.numCols());
		for (int k = 0; k < predictionMatrixSize; k++){
			SimpleMatrix predictionRow = conditionalPredictionMatrix.extractMatrix(k,k+1,0,conditionalPredictionMatrix.END);
			outPut.set(k, predictionRow.elementSum());
		}
		
		double normalizationFactor = 1/outPut.elementSum();
		outPut = outPut.scale(normalizationFactor);
		
		
		inpuMatrixBefore = inputMatrix;
		return null;
	}

}
