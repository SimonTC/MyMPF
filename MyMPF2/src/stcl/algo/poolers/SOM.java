package stcl.algo.poolers;

import java.util.Random;

import dk.stcl.core.som.SOM_SemiOnline;

public class SOM extends SOM_SemiOnline {

	public SOM(int columns, int rows, int inputLength, Random rand,
			double learningRate, double stddev, double activationCodingFactor) {
		super(columns, rows, inputLength, rand, learningRate, stddev,
				activationCodingFactor);
		// TODO Auto-generated constructor stub
	}

}
