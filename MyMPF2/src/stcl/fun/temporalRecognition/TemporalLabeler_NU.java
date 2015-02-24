package stcl.fun.temporalRecognition;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.NeoCorticalUnit;
import dk.stcl.core.basic.containers.SomNode;

public class TemporalLabeler_NU {

	/**
	 * labels rsom in temporal pooler in neocortical unit
	 * @param nu
	 * @param sequences
	 * @param sequenceLabels
	 */
	public void label(NeoCorticalUnit nu, ArrayList<SimpleMatrix[]> sequences, int[] sequenceLabels){
		assert sequences.size() == sequenceLabels.length : "The number of labels does not equal the number of sequences!";
		
		for (int sequenceID = 0; sequenceID < sequences.size(); sequenceID++){
			nu.flushTemporalMemory();
			SimpleMatrix[] sequence = sequences.get(sequenceID);
						
			for (SimpleMatrix m : sequence){
	    		SimpleMatrix ffOUtput = nu.feedForward(m);
	    		nu.feedBackward(ffOUtput);
			}
			
			SomNode bmu = nu.findTemporalBMU();
			bmu.setLabel(sequenceLabels[sequenceID]);
		}	


	}
}
