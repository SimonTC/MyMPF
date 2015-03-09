package stcl.fun.temporalRecognition;

import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.NU;
import dk.stcl.core.basic.containers.SomNode;

public class TemporalEvaluator_NU {

	public double evaluate(NU nu, ArrayList<SimpleMatrix[]> sequences, int[] sequenceLabels, SimpleMatrix joker, double noise, int iterations, Random rand){
		assert sequences.size() == sequenceLabels.length : "The number of labels does not equal the number of sequences!";
		
	    int curSeqID = 0;
	    int error = 0;
	    
	    for (int i = 0; i < iterations; i++){
	    	//Flush memory
	    	nu.flush();
	    	
	    	//Choose sequence	    	
	    	curSeqID = rand.nextInt(sequences.size());
	    	SimpleMatrix[] curSequence = sequences.get(curSeqID);
	    	SimpleMatrix ffOUtput;
	    	for (SimpleMatrix input : curSequence){
	    		if (rand.nextDouble() < noise)  input = joker;

	    		nu.feedForward(input);
	    		ffOUtput = nu.getFFOutput();
	    		nu.feedBackward(ffOUtput);    		
	    	}
	    	
	    	SomNode bmu = nu.getTemporalPooler().getRSOM().getBMU();
	    	int bmuLabel = bmu.getLabel();
	    	int correctLabel = sequenceLabels[curSeqID];
	    	if (bmuLabel != correctLabel) error++;
	    }
		
	    double fitness;
	    
	    if (error == 0){
	    	fitness = 1;
	    } else {
	    	fitness =  1.0 - (double)error / iterations;
	    }
	    
		return fitness;
	}

}
