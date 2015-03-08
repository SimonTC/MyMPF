package stcl.fun.temporalRecognition;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.NU;
import dk.stcl.core.basic.containers.SomNode;

public class TemporalLabeler_NU {

	public void label(NU nu, ArrayList<SimpleMatrix[]> sequences, int[] sequenceLabels, boolean all){
		if (all){
			labelAll(nu, sequences, sequenceLabels);
		} else {
			labelSingle(nu, sequences, sequenceLabels);
		}
	}
	
	/**
	 * labels rsom in temporal pooler in neocortical unit
	 * @param nu
	 * @param sequences
	 * @param sequenceLabels
	 */
	private void labelSingle(NU nu, ArrayList<SimpleMatrix[]> sequences, int[] sequenceLabels){
		assert sequences.size() == sequenceLabels.length : "The number of labels does not equal the number of sequences!";
		
		for (int sequenceID = 0; sequenceID < sequences.size(); sequenceID++){
			nu.flush();
			SimpleMatrix[] sequence = sequences.get(sequenceID);
						
			for (SimpleMatrix m : sequence){
	    		SimpleMatrix ffOUtput = nu.feedForward(m);
	    		nu.feedBackward(ffOUtput);
			}
			
			SomNode bmu = nu.getTemporalPooler().getRSOM().getBMU();
			bmu.setLabel(sequenceLabels[sequenceID]);
		}	
	}
	
	private void labelAll(NU nu, ArrayList<SimpleMatrix[]> sequences, int[] sequenceLabels){
		assert sequences.size() == sequenceLabels.length : "The number of labels does not equal the number of sequences!";
		
		//double[][] sequenceVotes = new double[nu.getTemporalPooler().getRSOM().getNodes().length][sequences.size()];
		int numModels = nu.getTemporalPooler().getRSOM().getNodes().length;
		int numSequences = sequences.size();
		
		SimpleMatrix sequenceVotes = new SimpleMatrix(numSequences, numModels);
		
		for (int sequenceID = 0; sequenceID < sequences.size(); sequenceID++){
			nu.flush();
			SimpleMatrix[] sequence = sequences.get(sequenceID);
			SimpleMatrix ffOUtput = null;		
			for (SimpleMatrix m : sequence){
	    		ffOUtput = nu.feedForward(m);
	    		nu.feedBackward(ffOUtput);
			}
			
			sequenceVotes.setRow(sequenceID, 0, ffOUtput.getMatrix().data);
		}	
		
		for (int model = 0; model < sequenceVotes.numCols(); model++){
			SimpleMatrix votes = sequenceVotes.extractVector(false, model);
			double max = Double.NEGATIVE_INFINITY;
			int maxID = -1;
			for (int seq = 0; seq < votes.getNumElements(); seq++){
				double v = votes.get(seq);
				if (v > max){
					max = v;
					maxID = seq;
				}
			}
			
			SomNode node = nu.getTemporalPooler().getRSOM().getNode(model);
			node.setLabel(sequenceLabels[maxID]);
			
		}
	}
}
