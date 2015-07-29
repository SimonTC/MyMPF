package stcl.algo.poolers;


import dk.stcl.core.som.SOM_SemiOnline;

public class SOM extends SOM_SemiOnline {
	private static final long serialVersionUID = 1L;
	public SOM(String s, int startLine){
		super(s, startLine);
	}
	public SOM(int mapSize, int inputLength, 
			double learningRate, double activationCodingFactor, double stdDev) {
		
		super(mapSize, inputLength, learningRate, activationCodingFactor, stdDev);
		
	}

	
}
