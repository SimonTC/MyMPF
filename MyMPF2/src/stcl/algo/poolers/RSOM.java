package stcl.algo.poolers;

import java.util.Random;

import dk.stcl.core.rsom.RSOM_SemiOnline;

public class RSOM extends RSOM_SemiOnline {

	public RSOM(int columns, int rows, int inputLength, Random rand,
			double learningRate, double stddev, double activationCodingFactor,
			double decayFactor) {
		super(columns, rows, inputLength, rand, learningRate, stddev,
				activationCodingFactor, decayFactor);
		// TODO Auto-generated constructor stub
	}

}
