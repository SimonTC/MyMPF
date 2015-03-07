package stcl.algo.brain;

import java.util.Observable;
import java.util.Observer;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.core.basic.containers.SomNode;
import stcl.algo.poolers.SOM;
import stcl.algo.poolers.Sequencer;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.algo.predictors.FirstOrderMM_Original;
import stcl.algo.predictors.FirstOrderPredictor;
import stcl.algo.predictors.Predictor;
import stcl.algo.predictors.Predictor_VOMM;
import stcl.algo.util.Normalizer;
import stcl.algo.util.Orthogonalizer;

public class NeoCorticalUnit3 implements NU{
	
	private SOM som;
	private Predictor_VOMM predictor;
	private SimpleMatrix biasMatrix;
	private double predictionBiasFactor;

	public NeoCorticalUnit3(int spatialMapSize, int inputLength, Random rand, int markovOrder, double predictionBiasFactor) {
		double spatialLearningRate = 0.1;
		double activationCodingFactor = 0.125;
		double stdDev = 2;//spatialMapSize; //Arbitrarily chosen
		som = new SOM(spatialMapSize, inputLength, rand, spatialLearningRate, activationCodingFactor, stdDev);
		
		double predictionLearningRate = 0.1;
		predictor = new Predictor_VOMM(markovOrder,predictionLearningRate, rand);
		
		this.predictionBiasFactor = predictionBiasFactor;

	}
	
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		
		//Spatial pooling
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
		
		return biasMatrix;
	}
	
	/**
	 * 
	 * @param inputMatrix
	 * @param correlationMatrix
	 * @return
	 */
	public SimpleMatrix feedBackward(SimpleMatrix inputMatrix){
		
		int predictionID = predictor.getNextPredictedSymbol();
		return som.getSomMap().get(predictionID).getVector();	
	}

	@Override
	public SOM getSOM() {
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
