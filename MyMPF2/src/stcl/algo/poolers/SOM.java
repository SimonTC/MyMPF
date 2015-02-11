package stcl.algo.poolers;

import java.util.Random;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane.MaximizeAction;

import dk.stcl.core.som.SOM_SemiOnline;
import dk.stcl.core.som.SOM_Simple;

public class SOM extends SOM_SemiOnline {

	public SOM(int mapSize, int inputLength, Random rand,
			double learningRate, double activationCodingFactor, double stdDev) {
		
		super(mapSize, inputLength, rand, learningRate, activationCodingFactor, stdDev);
		
	}

	
}
