/*
 * SOMTrainer.java
 *
 * Created on December 13, 2002, 2:37 PM
 */

package stcl.fun;


import java.util.Vector;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SpatialPooler;
import stcl.algo.som.SOMMap;
import stcl.algo.som.SomNode;
import stcl.graphics.MapRenderer;

/**
 *
 * @author  alanter
 */
public class SOMTrainer implements Runnable {
	// These constants can be changed to play with the learning algorithm
	private static final int	NUM_ITERATIONS = 500;
	
	private MapRenderer renderer;
	private SpatialPooler spatialPooler;
	private Vector<SimpleMatrix> inputs;
	private static boolean running;
	private Thread runner;
	
	/** Creates a new instance of SOMTrainer */
	public SOMTrainer() {
		running = false;
	}
	
	// Train the given lattice based on a vector of input vectors
	public void setTraining(SpatialPooler poolerToTrain, Vector<SimpleMatrix> in,
							MapRenderer latticeRenderer)
	{
		spatialPooler = poolerToTrain;
		inputs = in;
		renderer = latticeRenderer;
	}
	/**
	 * Train lattice without visualization
	 * @param latToTrain
	 * @param in
	 */
	public void setTraining(SpatialPooler poolerToTrain, Vector<SimpleMatrix> in){
		spatialPooler = poolerToTrain;
		inputs = in;
		renderer = null;
	}
	
	public void start() {
		run();
	}
	
	public void run() {
		int iteration = 0;
				
		while (iteration < NUM_ITERATIONS && running) {
			
			for (int i = 0; i < inputs.size(); i++){
				SimpleMatrix curInput = inputs.get(i);
				spatialPooler.feedForward(curInput);				
			}
			
			if (renderer != null){
				renderer.render(spatialPooler.getSOM(), iteration);
			}
			iteration++;
			spatialPooler.tick();
		}
		running = false;
	}

	public void stop() {
		if (runner != null) {
			running = false;
			while (runner.isAlive()) {};
		}
	}
}
