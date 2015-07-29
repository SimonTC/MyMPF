package stcl.algo.poolers;


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

}

