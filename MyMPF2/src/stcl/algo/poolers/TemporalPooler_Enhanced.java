package stcl.algo.poolers;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.predictors.FirstOrderPredictor;

public class TemporalPooler_Enhanced extends TemporalPooler {
	
	private boolean useMarkovPrediction;
	private FirstOrderPredictor predictor;
	private SimpleMatrix predictionMatrix;
	private double curPredictionLearningRate;

	public TemporalPooler_Enhanced(Random rand, int inputLength, int mapSize,
			double initialLearningRate, double stddev,
			double activationCodingFactor, double decay, boolean useMarkovPrediction) {
		super(rand, inputLength, mapSize, initialLearningRate, stddev,
				activationCodingFactor, decay);
		
		predictor = new FirstOrderPredictor(spatialMapSize);
		this.useMarkovPrediction = useMarkovPrediction;
		
	}
	
	@Override
	public SimpleMatrix feedForward(SimpleMatrix inputMatrix){
		if (useMarkovPrediction){
			predictionMatrix = predictor.predict(inputMatrix, curPredictionLearningRate);
			//predictionMatrix = predictor.predict(spatialFFOutputMatrixOrthogonalized, curPredictionLearningRate);
			/*
			if (DEBUG)System.out.println("Likelihood that SOM model ij will be the best to describe the next input");
			if (DEBUG)predictionMatrix.print();
			if (DEBUG)System.out.println();
			*/
		} 		
		
		//Transform spatial output matrix to vector
		double[] spatialFFOutputDataVector = spatialFFOutputMatrixOrthogonalized.getMatrix().data;		
		SimpleMatrix temporalFFInputVector = new SimpleMatrix(1, spatialFFOutputDataVector.length);
		temporalFFInputVector.getMatrix().data = spatialFFOutputDataVector;
		
		//Temporal classification
		SimpleMatrix temporalFFOutputMatrix = temporalPooler.feedForward(temporalFFInputVector);
		
		ffOutput = temporalFFOutputMatrix;
		
		
	}

}
