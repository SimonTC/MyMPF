package stcl.fun.temporalRecognition;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import dk.stcl.core.basic.containers.SomNode;
import dk.stcl.core.rsom.IRSOM;
import dk.stcl.core.som.ISOM;

/**
 * This class is used for labeling the nodes of a SOM
 * @author Simon
 *
 */
public class RSomLabeler {
	
	/**
	 *  labels a single rsom
	 * @param rsom
	 * @param sequences
	 * @param sequenceLabels
	 */
	public void label (IRSOM  rsom, ArrayList<SimpleMatrix[]> sequences, int[] sequenceLabels){
		assert sequences.size() == sequenceLabels.length : "The number of labels does not equal the number of sequences!";
		
		for (int sequenceID = 0; sequenceID < sequences.size(); sequenceID++){
			rsom.flush();
			SimpleMatrix[] sequence = sequences.get(sequenceID);
						
			for (SimpleMatrix m : sequence){
				rsom.step(m);
			}
			
			SomNode bmu = rsom.getBMU();
			bmu.setLabel(sequenceLabels[sequenceID]);
		}	
	}
	
	/**
	 * labels rsom in som-rsom pair
	 * @param som
	 * @param rsom
	 * @param sequences
	 * @param sequenceLabels
	 */
	public void label(ISOM som, IRSOM rsom, ArrayList<SimpleMatrix[]> sequences, int[] sequenceLabels){
		assert sequences.size() == sequenceLabels.length : "The number of labels does not equal the number of sequences!";
		
		for (int sequenceID = 0; sequenceID < sequences.size(); sequenceID++){
			rsom.flush();
			SimpleMatrix[] sequence = sequences.get(sequenceID);
						
			for (SimpleMatrix m : sequence){
	    		//Spatial classification
	    		som.step(m);
	    		SimpleMatrix spatialFFOutputMatrix = som.computeActivationMatrix();
	    		
	    		spatialFFOutputMatrix = orthogonalize(spatialFFOutputMatrix);
	    		
	    		//Transform spatial output matrix to vector
	    		SimpleMatrix temporalFFInputVector = new SimpleMatrix(spatialFFOutputMatrix);
	    		temporalFFInputVector.reshape(1, spatialFFOutputMatrix.getMatrix().data.length);
	    		
	    		//Temporal classification
	    		rsom.step(temporalFFInputVector);

			}
			
			SomNode bmu = rsom.getBMU();
			bmu.setLabel(sequenceLabels[sequenceID]);
		}	

	}
	
	/**
	 * labels rsom in temporal pooler in spatial pooler - temporal pooler pair
	 * @param spatialPooler
	 * @param temporalPooler
	 * @param sequences
	 * @param sequenceLabels
	 */
	public void label(SpatialPooler spatialPooler, TemporalPooler temporalPooler, ArrayList<SimpleMatrix[]> sequences, int[] sequenceLabels){
		assert sequences.size() == sequenceLabels.length : "The number of labels does not equal the number of sequences!";
		
		for (int sequenceID = 0; sequenceID < sequences.size(); sequenceID++){
			temporalPooler.flushTemporalMemory();
			SimpleMatrix[] sequence = sequences.get(sequenceID);
						
			for (SimpleMatrix m : sequence){
	    		//Spatial classification
	    		SimpleMatrix spatialFFOutputMatrix = spatialPooler.feedForward(m);
	    		
	    		//Transform spatial output matrix to vector
	    		SimpleMatrix temporalFFInputVector = new SimpleMatrix(spatialFFOutputMatrix);
	    		temporalFFInputVector.reshape(1, spatialFFOutputMatrix.getMatrix().data.length);
	    		
	    		//Temporal classification
	    		temporalPooler.feedForward(temporalFFInputVector);

			}
			
			SomNode bmu = temporalPooler.getRSOM().getBMU();
			bmu.setLabel(sequenceLabels[sequenceID]);
		}	

	}

	
	private SimpleMatrix orthogonalize(SimpleMatrix m) {
		double max = Double.NEGATIVE_INFINITY;
		int maxID = -1;
		for (int i = 0; i < m.getNumElements(); i++){
			double value = m.get(i);
			if (value > max){
				max = value;
				maxID = i;
			}
		}
		
		SimpleMatrix ortho = new SimpleMatrix(m.numRows(), m.numCols());
		ortho.set(maxID, 1);
		return ortho;
	}


}
