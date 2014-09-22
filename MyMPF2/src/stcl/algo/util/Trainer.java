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
		double leakyCoefficient = 0.3;
		
		
		NeoCorticalUnit nu = new NeoCorticalUnit(rand, maxIterations, ffInputLength, spatialMapSize, temporalMapSize, biasInfluence, predictionLearningRate, useMarkovPrediction, leakyCoefficient);
		double[][] trainingData = {{0,1,0,0,1,1,0,1,0,1}, 
						  		   {0,1,1,1,0,1,0,0,0,1}};
		
		double SQME = 1;
		SimpleMatrix ffOutput = null;
		SimpleMatrix fbOutput = null;
		int i = 0;
		do {
			
			for (int sample = 0; sample < trainingData.length; sample++){
				double[][] input = {trainingData[sample]};
				SQME = 0;
				SimpleMatrix inputVector = new SimpleMatrix(input);
				ffOutput = nu.feedForward(inputVector);
				
				fbOutput = nu.feedBackward(ffOutput, ffOutput);
				
				double[] actual = fbOutput.getMatrix().data;
				int j = 0;
				if ( i == 0) j = 1;
				if ( i == 1) j = 0;
				SQME += calculateSQME(trainingData[j], actual);
				
			}
			
			
			System.out.println("Step " + i++ + "SQME: " + SQME);
			
		} while (SQME > 0.001);
		
		System.out.println();
		
		
		/*
		
		System.out.println("FF Output:");
		ffOutput.print(5, 3);
		System.out.println();
		
		System.out.println("FB Output");
		fbOutput.print(5, 3);
		System.out.println();
		
		System.out.println("Given input:");
		inputVector.print(5, 3);
		*/
		
		System.out.println("Finished");

	}
	
	private static double calculateSQME(double[] expected, double[] actual){
		double totalError = 0;
		for (int i = 0; i < expected.length; i++){
			double error = expected[i] - actual[i];
			error *= error;
			totalError += error;
		}
		
		return totalError / expected.length;
	}

}
