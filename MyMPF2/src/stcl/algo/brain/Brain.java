package stcl.algo.brain;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class Brain {
	
	NeoCorticalUnit nu1;
	NeoCorticalUnit nu2;
	
	Random rand;
	int maxIterations;
	int ffInputLength1;
	int ffInputLength2;
	int spatialMapSize1;
	int spatialMapSize2;
	int temporalMapSize1;
	int temporalMapSize2;
	
	double biasInfluence;
	double predictionLearningRate;
	boolean useMarkovPrediction;
	double leakyCoefficient;
	
	public Brain() {
		rand = new Random();
		maxIterations = 100;
		ffInputLength1 = 16 * 16;
		spatialMapSize1 = 5;
		temporalMapSize1 = 5;
		ffInputLength2 = temporalMapSize1 * temporalMapSize1;
		temporalMapSize2 = 5;
		
		biasInfluence = 0.1;
		predictionLearningRate = 0.6;
		useMarkovPrediction = true;
		leakyCoefficient = 0.1;
		
		nu1 = new NeoCorticalUnit(rand, maxIterations, ffInputLength1, spatialMapSize1, temporalMapSize1, biasInfluence, predictionLearningRate, useMarkovPrediction, leakyCoefficient);
		
		nu2 = new NeoCorticalUnit(rand, maxIterations, ffInputLength2, spatialMapSize2, temporalMapSize2, biasInfluence, predictionLearningRate, useMarkovPrediction, leakyCoefficient);
	}
	
	public void step(SimpleMatrix inputVector){
		//Feed input vector forward
		
		//Collect ff output from unit one
		
		//Do reward correlation
		
		//Send unit one's output to unit 2
		
		//Give some fb input to unit 2
		
		//Collect fb output from unit 2
		
		//Give FB output from unit 2 to unit 1
		
		//Collect FB output from unit 1
		
		//Return something
		
		
	}

}
