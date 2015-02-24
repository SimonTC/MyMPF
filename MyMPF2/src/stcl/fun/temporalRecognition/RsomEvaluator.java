package stcl.fun.temporalRecognition;

import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.core.basic.containers.SomNode;
import dk.stcl.core.rsom.IRSOM;
import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;

public class RsomEvaluator {
	
	public double evaluate(SpatialPooler spatialPooler, TemporalPooler temporalPooler, ArrayList<SimpleMatrix[]> sequences, int[] sequenceLabels, SimpleMatrix joker, double noise, int iterations, Random rand){
		assert sequences.size() == sequenceLabels.length : "The number of labels does not equal the number of sequences!";
		
	    int curSeqID = 0;
	    int error = 0;
	    
	    for (int i = 0; i < iterations; i++){
	    	//Flush memory
	    	temporalPooler.flushTemporalMemory();
	    	
	    	//Choose sequence	    	
	    	curSeqID = rand.nextInt(sequences.size());
	    	SimpleMatrix[] curSequence = sequences.get(curSeqID);
	    	
	    	for (SimpleMatrix input : curSequence){
	    		if (rand.nextDouble() < noise) input = joker;
	    		//Spatial classification
	    		SimpleMatrix spatialFFOutputMatrix = spatialPooler.feedForward(input);
	    		
	    		//Transform spatial output matrix to vector
	    		SimpleMatrix temporalFFInputVector = new SimpleMatrix(spatialFFOutputMatrix);
	    		temporalFFInputVector.reshape(1, spatialFFOutputMatrix.getMatrix().data.length);
	    		
	    		//Temporal classification
	    		temporalPooler.feedForward(temporalFFInputVector);	    		
	    	}
	    	
	    	SomNode bmu = temporalPooler.getRSOM().getBMU();
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
	
	public double evaluate(IRSOM  rsom, ArrayList<SimpleMatrix[]> sequences, int[] sequenceLabels, SimpleMatrix joker, double noise, int iterations, Random rand){
		assert sequences.size() == sequenceLabels.length : "The number of labels does not equal the number of sequences!";
		
	    int curSeqID = 0;
	    int error = 0;
	    
	    for (int i = 0; i < iterations; i++){
	    	//Flush memory
	    	rsom.flush();
	    	
	    	//Choose sequence	    	
	    	curSeqID = rand.nextInt(sequences.size());
	    	SimpleMatrix[] curSequence = sequences.get(curSeqID);
	    	
	    	for (SimpleMatrix input : curSequence){
	    		if (rand.nextDouble() < noise) input = joker;
	    		rsom.step(input);    		
	    	}
	    	
	    	SomNode bmu = rsom.getBMU();
	    	if (bmu.getLabel() != sequenceLabels[curSeqID]) error++;
	    }
		
	    double fitness;
	    
	    if (error == 0){
	    	fitness = 1;
	    } else {
	    	fitness = (double) 1 - error / iterations;
	    }
	    
		return fitness;
	}
}
