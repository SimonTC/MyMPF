package stcl.algo.poolers;

import java.util.Random;

import dk.stcl.core.rsom.RSOM_SemiOnline;
import dk.stcl.core.rsom.RSOM_Simple;

public class RSOM extends RSOM_Simple {

	public RSOM(int mapSize, int inputLength, Random rand, int maxIterations, 
			double learningRate, double activationCodingFactor,
			double decayFactor) {
		super(mapSize, inputLength, rand, maxIterations, learningRate,
				activationCodingFactor, decayFactor);
		// TODO Auto-generated constructor stub
	}

}

