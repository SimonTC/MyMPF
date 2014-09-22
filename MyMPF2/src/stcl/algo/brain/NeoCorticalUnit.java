package stcl.algo.brain;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.biasunits.BiasUnit;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.algo.predictors.FirstOrderPredictor;

public class NeoCorticalUnit {
	
	private SpatialPooler spatialPooler;
	private TemporalPooler temporalPooler;
	private BiasUnit biasUnit;
	private FirstOrderPredictor predictor;
	private SimpleMatrix biasMatrix;
	private SimpleMatrix predictionMatrix;
	
	private int ffInputVectorSize;
	private int spatialMapSize;
	private int temporalMapSize;
	
	//Learning rates
	private double predictionLearningRate; //TODO: Find correct name for it
	private double curNoiseMagnitude;
	//private double biasInfluence;  
	
	private boolean useMarkovPrediction;
	
	public NeoCorticalUnit(Random rand, int maxIterations, int ffInputLength, int spatialMapSize, int temporalMapSize, double biasInfluence, double predictionLearningRate, boolean useMarkovPrediction) {
		//TODO: All parameters should be handled in parameter file
		spatialPooler = new SpatialPooler(rand, maxIterations, ffInputLength, spatialMapSize);
		temporalPooler = new TemporalPooler(rand, maxIterations, spatialMapSize * spatialMapSize, temporalMapSize);
		biasUnit = new BiasUnit(ffInputLength, biasInfluence, rand);
		predictor = new FirstOrderPredictor(spatialMapSize);
		biasMatrix = new SimpleMatrix(spatialMapSize, spatialMapSize);
		biasMatrix.set(1);
		ffInputVectorSize = ffInputLength;
		this.predictionLearningRate = predictionLearningRate;
		this.useMarkovPrediction = useMarkovPrediction;
		this.spatialMapSize = spatialMapSize;
		this.temporalMapSize = temporalMapSize;
	}
	
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		
		//Spatial classification
		SimpleMatrix spatialFFOutputMatrix = spatialPooler.feedForward(inputVector);
		
		//Bias output by the prediction from t-1
		SimpleMatrix spatialFFOutputMatrixBiased = spatialFFOutputMatrix.elementMult(biasMatrix);	
		
		//Normalize output
		double sum = spatialFFOutputMatrixBiased.elementSum();
		spatialFFOutputMatrixBiased = spatialFFOutputMatrixBiased.scale(1/sum);
		
		//Predict next input
		if (useMarkovPrediction){
			predictionMatrix = predictor.predict(spatialFFOutputMatrixBiased, predictionLearningRate);
		} 
		
		//Transform spatial output matrix to vector
		double[] spatialFFOutputDataVector = spatialFFOutputMatrixBiased.getMatrix().data;		
		SimpleMatrix temporalFFInputVector = new SimpleMatrix(1, spatialFFOutputDataVector.length);
		temporalFFInputVector.getMatrix().data = spatialFFOutputDataVector;
		
		//Temporal classification
		SimpleMatrix temporalFFOutputMatrix = temporalPooler.feedForward(temporalFFInputVector);
		
		return temporalFFOutputMatrix;
	}
	
	public SimpleMatrix feedBackward(SimpleMatrix inputMatrix, SimpleMatrix correlationMatrix){
		
		//Selection of best temporal model
		SimpleMatrix temporalPoolerFBOutput = temporalPooler.feedBackward(inputMatrix);
		
		//Transformation into matrix
		temporalPoolerFBOutput.reshape(spatialMapSize, spatialMapSize);
		
		//Combine FB output from temporal pooler with bias and prediction (if enabled)
		biasMatrix = temporalPoolerFBOutput;
		if (useMarkovPrediction){
			biasMatrix = biasMatrix.elementMult(predictionMatrix);
		} 
		
		//Selection of best spatial mode
		SimpleMatrix spatialPoolerFBOutputVector = spatialPooler.feedBackward(biasMatrix);
		
		return spatialPoolerFBOutputVector;
	}

}
