package stcl.algo.util;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.NeoCorticalUnit;

public class Trainer {

	public Trainer() {
		// TODO Auto-generated constructor stub
	}

	public static void main(String[] args) {
		Random rand = new Random();
		int maxIterations = 100;
		int ffInputLength = 10;
		int spatialMapSize = 6;
		int temporalMapSize = 6;
		double biasInfluence = 0.1;
		double predictionLearningRate = 0.25;
		boolean useMarkovPrediction = true;
		
		
		NeoCorticalUnit nu = new NeoCorticalUnit(rand, maxIterations, ffInputLength, spatialMapSize, temporalMapSize, biasInfluence, predictionLearningRate, useMarkovPrediction);
		//double[][] data = {{0,1,0,0,1,1,0,1,0,1}};
		double[][] data = {{10,5,3,2,4,9,6,7,1,2}};
		SimpleMatrix inputVector = new SimpleMatrix(data);
		
		SimpleMatrix ffOutput = nu.feedForward(inputVector);
		
		SimpleMatrix fbOutput = nu.feedBackward(ffOutput, ffOutput);
		
		System.out.println("FF Output:");
		ffOutput.print(5, 3);
		System.out.println();
		
		System.out.println("FB Output");
		fbOutput.print(5, 3);
		System.out.println();
		
		System.out.println("Given input:");
		inputVector.print(5, 3);
		
		System.out.println("Finished");

	}

}
