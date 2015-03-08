package stcl.algo.poolers;

import java.util.Random;

import dk.stcl.core.rsom.RSOM_SemiOnline;

public class RSOM extends RSOM_SemiOnline {

	public RSOM(int mapSize, int inputLength, Random rand,
			double learningRate, double activationCodingFactor, double stdDev,
			double decayFactor) {
		super(mapSize, inputLength, rand, learningRate, activationCodingFactor, stdDev, decayFactor);
	}

}

