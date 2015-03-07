package stcl.algo.brain;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SOM;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.algo.predictors.Predictor_VOMM;
import stcl.algo.util.Normalizer;

public class SOM_VOMM implements NU {
	private SOM som;
	private Predictor_VOMM predictor;
	private SimpleMatrix biasMatrix;
	private double predictionBiasFactor;
	private SimpleMatrix predictedWeights;
	
	public SOM_VOMM(int spatialMapSize, int inputLength, Random rand, int markovOrder, double predictionBiasFactor) {
		double spatialLearningRate = 0.1;
		double activationCodingFactor = 0.125;
		double stdDev = 2;//spatialMapSize; //Arbitrarily chosen
		
		double predictionLearningRate = 0.1;
		predictor = new Predictor_VOMM(markovOrder,predictionLearningRate, rand);
		som = new SOM(spatialMapSize, inputLength, rand, spatialLearningRate, activationCodingFactor, stdDev);
		
		
		
		
		this.predictionBiasFactor = predictionBiasFactor;

	}
	
	
	@Override
	public SimpleMatrix feedForward(SimpleMatrix inputVector) {
		
		som.step(inputVector);
		SimpleMatrix spatialOutput = som.computeActivationMatrix();
		
		//Normalize
		spatialOutput = Normalizer.normalize(spatialOutput);
		
		//Bias
		SimpleMatrix biasedOutput = spatialOutput;
		if (biasMatrix!= null){
			biasedOutput = spatialOutput.plus(predictionBiasFactor, biasMatrix);
		}
	
		biasedOutput = Normalizer.normalize(biasedOutput);
		
		//Predict
		biasMatrix = predictor.predict(biasedOutput, 0.1, true);
		
		int predictionID = predictor.getNextPredictedSymbol();
		predictedWeights = som.getSomMap().get(predictionID).getVector();	
		
		return predictedWeights;
	}

	@Override
	public SimpleMatrix feedBackward(SimpleMatrix inputMatrix) {
		return predictedWeights;
	}
	
	public SOM getSOM(){
		return som;
	}
	
	@Override
	public void printModel() {
		predictor.printModel();
		
	}


	@Override
	public SpatialPooler getSpatialPooler() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public TemporalPooler getTemporalPooler() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public SimpleMatrix getFfOutput() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public SimpleMatrix getFbOutput() {
		// TODO Auto-generated method stub
		return null;
	}

}
