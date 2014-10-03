package stcl.graphics;

import java.awt.Dimension;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import stcl.graphics.MapDrawerGRAY.MapPanel;

public abstract class MapDrawer extends JFrame {

	public abstract void updateMap(SimpleMatrix dataMap);
}
