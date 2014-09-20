package stcl.fun;

import java.util.Vector;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SpatialPooler;
import stcl.algo.som.SOMMap;


public class MyTrainer {
	// These constants can be changed to play with the learning algorithm
		private static final int	NUM_ITERATIONS = 500;
		
		// These two depend on the size of the lattice, so are set later
		private Vector<SimpleMatrix> inputs;
		private SpatialPooler spatialPooler;
			
		
		/**
		 * Train lattice without visualization
		 * @param latToTrain
		 * @param in
		 */
		public void setTraining(SpatialPooler pooler, Vector<SimpleMatrix> in){
			spatialPooler = pooler;
			inputs = in;
		}
		
		public void run() {
			int iteration = 0;
					
			while (iteration < NUM_ITERATIONS ) {
				
				for (int i = 0; i < inputs.size(); i++){
					SimpleMatrix curInput = inputs.get(i);
					spatialPooler.feedForward(curInput);				
				}		
				
				iteration++;
				spatialPooler.tick();
			}
			
		}
}
