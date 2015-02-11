package stcl.algo.poolers;

import java.util.Random;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane.MaximizeAction;

import dk.stcl.core.som.SOM_SemiOnline;
import dk.stcl.core.som.SOM_Simple;

public class SOM extends SOM_Simple {

	public SOM(int mapSize, int inputLength, Random rand, int maxIterations,
			double learningRate, double activationCodingFactor) {
		
		super(mapSize, inputLength, rand, maxIterations, learningRate,
				activationCodingFactor);
		// TODO Auto-generated constructor stub
	}

	
}
