/*
 * SOMDemo.java
 *
 * Created on December 13, 2002, 2:31 PM
 */

package stcl.fun;


import java.awt.image.BufferedImage;
import java.util.Random;
import java.util.Vector;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SpatialPooler;
import stcl.algo.som.SOMMap;
import stcl.algo.util.Trainer;
import stcl.graphics.MapRenderer;

/**
 *
 * @author  alanter
 */
public class SOMDemoAppNoVisual {
	int size = 3;
	//private SOMTrainer trainer;
	private MySOMTrainer trainer;
	private SOMMap lattice;
	private Vector<SimpleMatrix> inputVectors;
	private SpatialPooler pooler;
	
	/** Creates new form SOMDemo */
	public SOMDemoAppNoVisual() {

		SimpleMatrix tempVec;
		pooler = new SpatialPooler(new Random(), 500, 3, size);
		lattice = pooler.getSOM();

		trainer = new MySOMTrainer();
		inputVectors = new Vector<SimpleMatrix>();

		// Make some colors.  Red, Green, Blue, Yellow, Purple, Black,
		// white, and gray
		tempVec = new SimpleMatrix(1,3);
		tempVec.set(0, 1);
		tempVec.set(1, 0);
		tempVec.set(2, 0);
		inputVectors.addElement(tempVec);
		tempVec = new SimpleMatrix(1,3);
		tempVec.set(0, 0);
		tempVec.set(1, 1);
		tempVec.set(2, 0);
		inputVectors.addElement(tempVec);
		tempVec = new SimpleMatrix(1,3);
		tempVec.set(0, 0);
		tempVec.set(1, 0);
		tempVec.set(2, 1);
		inputVectors.addElement(tempVec);
		tempVec = new SimpleMatrix(1,3);
		tempVec.set(0, 1);
		tempVec.set(1, 1);
		tempVec.set(2, 0);
		inputVectors.addElement(tempVec);
		tempVec = new SimpleMatrix(1,3);
		tempVec.set(0, 1);
		tempVec.set(1, 0);
		tempVec.set(2, 1);
		inputVectors.addElement(tempVec);
		tempVec = new SimpleMatrix(1,3);
		tempVec.set(0, 0);
		tempVec.set(1, 1);
		tempVec.set(2, 1);
		inputVectors.addElement(tempVec);
		tempVec = new SimpleMatrix(1,3);
		tempVec.set(0, 0);
		tempVec.set(1, 0);
		tempVec.set(2, 0);
		inputVectors.addElement(tempVec);
		tempVec = new SimpleMatrix(1,3);
		tempVec.set(0, 1);
		tempVec.set(1, 1);
		tempVec.set(2, 1);
		inputVectors.addElement(tempVec);
		tempVec = new SimpleMatrix(1,3);
		tempVec.set(0, 0.5);
		tempVec.set(1, 0.5);
		tempVec.set(2, 0.5);
		inputVectors.addElement(tempVec);

/*		// Make it four shades of red.
		tempVec = new SOMVector();
		tempVec.addElement(new Double(1));
		tempVec.addElement(new Double(0));
		tempVec.addElement(new Double(0));
		inputVectors.addElement(tempVec);
		tempVec = new SOMVector();
		tempVec.addElement(new Double(0.7));
		tempVec.addElement(new Double(0));
		tempVec.addElement(new Double(0));
		inputVectors.addElement(tempVec);
		tempVec = new SOMVector();
		tempVec.addElement(new Double(0.4));
		tempVec.addElement(new Double(0));
		tempVec.addElement(new Double(0));
		inputVectors.addElement(tempVec);
		tempVec = new SOMVector();
		tempVec.addElement(new Double(0.1));
		tempVec.addElement(new Double(0));
		tempVec.addElement(new Double(0));
		inputVectors.addElement(tempVec);
		tempVec = new SOMVector();
		tempVec.addElement(new Double(0));
		tempVec.addElement(new Double(0));
		tempVec.addElement(new Double(0));
		inputVectors.addElement(tempVec);
		*/
	}

	/**
	 * @param args the command line arguments
	 */
	public static void main(String args[]) {
		SOMDemoAppNoVisual theApp = new SOMDemoAppNoVisual();
		
		theApp.go();
		//new SOMDemoApp().show();
	}

	public void go() {

		trainer.setTraining(pooler, inputVectors);
		trainer.start();
	}
	


}
