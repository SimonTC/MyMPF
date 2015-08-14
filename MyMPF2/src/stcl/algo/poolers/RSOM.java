package stcl.algo.poolers;


import java.util.Random;

import dk.stcl.core.rsom.RSOM_SemiOnline;

public class RSOM extends RSOM_SemiOnline {

	public RSOM(String s, int startLine){
		super(s, startLine);
	}
	
	public RSOM(int mapSize, int inputLength, 
			double learningRate, double activationCodingFactor, double stdDev,
			double decayFactor) {
		super(mapSize, inputLength,  learningRate, activationCodingFactor, stdDev, decayFactor);
	}
	
	public RSOM(int mapSize, int inputLength, Random rand,
			double learningRate, double activationCodingFactor, double stdDev,
			double decayFactor) {
		super(mapSize, inputLength,  rand, learningRate, activationCodingFactor, stdDev, decayFactor);
	}

}

