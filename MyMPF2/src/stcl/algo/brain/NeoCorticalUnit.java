package stcl.algo.brain;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.biasunits.Bias;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.algo.predictors.FirstOrderPredictor;

public class NeoCorticalUnit {
	
	private SpatialPooler spatialPooler;
	private TemporalPooler temporalPooler;
	private Bias biasUnit;
	private FirstOrderPredictor predictor;
	private SimpleMatrix predictionMatrix;
	
	private int ffInputVectorSize;
	
	//Learning rates
	private double predictionLearningRate; //TODO: Find correct name for it
	private double curNoiseMagnitude;
	//private double biasInfluence;
	
	private boolean biasActions;
	
	public NeoCorticalUnit(Random rand, int maxIterations, int ffInputLength, int spatialMapSize, int temporalMapSize, double biasInfluence, double predictionLearningRate) {
		//TODO: All parameters should be handled in parameter file
		spatialPooler = new SpatialPooler(rand, maxIterations, ffInputLength, spatialMapSize);
		temporalPooler = new TemporalPooler(rand, maxIterations, ffInputLength * ffInputLength, temporalMapSize);
		biasUnit = new Bias(ffInputLength, biasInfluence, rand);
		predictor = new FirstOrderPredictor(ffInputLength);
		ffInputVectorSize = ffInputLength;
		this.predictionLearningRate = predictionLearningRate;
	}
	
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		
		//Spatial classification
		SimpleMatrix spatialFFOutputMatrix = spatialPooler.feedForward(inputVector);
		
		//Bias output
		SimpleMatrix biasedSpatialFFOutputMatrix;
		if (biasActions){
			biasedSpatialFFOutputMatrix = biasUnit.biasSpatialFFOutput(spatialFFOutputMatrix);
		} else {
			biasedSpatialFFOutputMatrix = spatialFFOutputMatrix;
		}
		
		//Predict next input
		predictionMatrix = predictor.predict(biasedSpatialFFOutputMatrix, predictionLearningRate);
		
		//Transform spatial output matrix to vector
		double[] spatialFFOutputDataVector = biasedSpatialFFOutputMatrix.getMatrix().data;		
		SimpleMatrix temporalFFInputVector = new SimpleMatrix(1, spatialFFOutputDataVector.length);
		temporalFFInputVector.getMatrix().data = spatialFFOutputDataVector;
		
		//Temporal classification
		SimpleMatrix temporalFFOutputMatrix = temporalPooler.feedForward(temporalFFInputVector);
		
		return temporalFFOutputMatrix;
	}
	
	public SimpleMatrix feedBackward(SimpleMatrix inputMatrix, SimpleMatrix correlationMatrix){
		
		//Selection of best temporal model
		SimpleMatrix temporalPoolerFBOutputVector = temporalPooler.feedBack(inputMatrix);
		
		//Transformation into matrix
		SimpleMatrix temporalFBOutputMatrix = new SimpleMatrix(ffInputVectorSize, ffInputVectorSize);
		temporalFBOutputMatrix.getMatrix().data = temporalPoolerFBOutputVector.getMatrix().data;
		
		//Combine FB output from temporal pooler with prediction
		SimpleMatrix biasedFBTemporalOutputMatrix = biasUnit.calculateBias(temporalFBOutputMatrix, correlationMatrix, curNoiseMagnitude);
		
		//Selection of best spatial mode
		SimpleMatrix spatialPoolerFBOutputVector = spatialPooler.feedBack(biasedFBTemporalOutputMatrix);
		
		return spatialPoolerFBOutputVector;
	}

}
