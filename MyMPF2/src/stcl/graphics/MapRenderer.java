package stcl.graphics;

import java.awt.LayoutManager;

import javax.swing.JPanel;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.som.SOM;

public abstract class MapRenderer extends JPanel {

	public MapRenderer() {
		// TODO Auto-generated constructor stub
	}

	public MapRenderer(LayoutManager layout) {
		super(layout);
		// TODO Auto-generated constructor stub
	}

	public MapRenderer(boolean isDoubleBuffered) {
		super(isDoubleBuffered);
		// TODO Auto-generated constructor stub
	}

	public MapRenderer(LayoutManager layout, boolean isDoubleBuffered) {
		super(layout, isDoubleBuffered);
		// TODO Auto-generated constructor stub
	}
	
	public abstract void render(SOM lattice, int iteration);
	
	public abstract void render(SimpleMatrix matrix, int iteration);

}
