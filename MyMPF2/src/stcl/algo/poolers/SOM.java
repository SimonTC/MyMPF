package stcl.algo.poolers;

import java.util.Random;

import dk.stcl.core.som.SOM_SemiOnline;

public class SOM extends SOM_SemiOnline {
	private static final long serialVersionUID = 1L;
	public SOM(String s){
		super(s);
	}
	public SOM(int mapSize, int inputLength, Random rand,
			double learningRate, double activationCodingFactor, double stdDev) {
		
		super(mapSize, inputLength, rand, learningRate, activationCodingFactor, stdDev);
		
	}

	
}
