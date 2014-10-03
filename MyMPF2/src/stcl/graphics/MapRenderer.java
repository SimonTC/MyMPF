package stcl.graphics;

import java.awt.LayoutManager;

import javax.swing.JPanel;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.som.SOM;

public abstract class MapRenderer extends JPanel {

	public abstract void render(SOM lattice, int iteration);
	
	public abstract void render(SimpleMatrix matrix, int iteration);

}
