package stcl.graphics;

import javax.swing.JPanel;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.som.som.SOM;

public abstract class MapRenderer extends JPanel {

	public abstract void render(SOM lattice, int iteration);
	
	public abstract void render(SimpleMatrix matrix, int iteration);

}
