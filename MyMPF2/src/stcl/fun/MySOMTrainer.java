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
public class MySOMTrainer implements Runnable {
	// These constants can be changed to play with the learning algorithm
	private static final int	NUM_ITERATIONS = 500;
	
	private MapRenderer renderer;
	private SpatialPooler spatialPooler;
	private Vector<SimpleMatrix> inputs;
	private static boolean running;
	private Thread runner;
	
	/** Creates a new instance of SOMTrainer */
	public MySOMTrainer() {
		running = false;
	}
	
	// Train the given lattice based on a vector of input vectors
	public void setTraining(SpatialPooler poolerToTrain, Vector<SimpleMatrix> in,
							MapRenderer mapRenderer)
	{
		spatialPooler = poolerToTrain;
		inputs = in;
		renderer = mapRenderer;
	}
	
	
	public void start() {
		if (spatialPooler != null) {
			runner = new Thread(this);
			runner.setPriority(Thread.MIN_PRIORITY);
			running = true;
			runner.start();
		}
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
			System.out.println(iteration + " Checksum: " + spatialPooler.getSOM().getErrorMatrix().elementSum());
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
