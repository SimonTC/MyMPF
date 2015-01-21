package stcl.graphics;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

public abstract class MapDrawer extends JFrame {

	public abstract void updateMap(SimpleMatrix dataMap);
}
