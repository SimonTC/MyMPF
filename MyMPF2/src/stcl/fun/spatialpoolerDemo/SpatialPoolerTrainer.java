/*
 * SOMTrainer.java
 *
 * Created on December 13, 2002, 2:37 PM
 */

package stcl.fun.spatialpoolerDemo;


import java.util.Vector;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SpatialPooler;
import stcl.algo.som.SOM;
import stcl.algo.som.SomNode;
import stcl.graphics.RGBMapRenderer;

/**
 *
 * @author  alanter
 */
public class SpatialPoolerTrainer implements Runnable {
	// These constants can be changed to play with the learning algorithm
	private static final int	NUM_ITERATIONS = 500;
	
	private RGBMapRenderer renderer;
	private SpatialPooler spatialPooler;
	private Vector<SimpleMatrix> inputs;
	private static boolean running;
	private Thread runner;
	
	/** Creates a new instance of SOMTrainer */
	public SpatialPoolerTrainer() {
		running = false;
	}
	
	// Train the given lattice based on a vector of input vectors
	public void setTraining(SpatialPooler poolerToTrain, Vector<SimpleMatrix> in,
							RGBMapRenderer mapRenderer)
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
