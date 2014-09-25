/*
 * SOMTrainer.java
 *
 * Created on December 13, 2002, 2:37 PM
 */

package stcl.fun.somdemo;


import java.util.Vector;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SpatialPooler;
import stcl.algo.som.SOM;
import stcl.algo.som.SomNode;
import stcl.graphics.MapRenderer;

/**
 *
 * @author  alanter
 */
public class SOMTrainer implements Runnable {
	// These constants can be changed to play with the learning algorithm
	private static final int	NUM_ITERATIONS = 500;
	
	private SOM map;
	private MapRenderer renderer;
	private Vector<SimpleMatrix> inputs;
	private static boolean running;
	private Thread runner;
	
	/** Creates a new instance of SOMTrainer */
	public SOMTrainer() {
		running = false;
	}
	
	// Train the given lattice based on a vector of input vectors
	public void setTraining(SOM map, Vector<SimpleMatrix> in,
							MapRenderer mapRenderer)
	{
		this.map = map;
		inputs = in;
		renderer = mapRenderer;
	}
	
	
	public void start() {
		if (map != null) {
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
				double learningRate = 0.7;
				double initialRadius = map.getHeight() / 2;
				double neighborHoodRadius  = initialRadius * Math.exp(-iteration/initialRadius);
				
				
				map.step(curInput, learningRate, neighborHoodRadius);
								
			}
			
			if (renderer != null){
				renderer.render(map, iteration);
			}
			iteration++;

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
