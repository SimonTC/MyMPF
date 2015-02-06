package stcl.graphics;

import java.awt.Color;
import java.awt.Graphics;

import javax.swing.JPanel;

import org.ejml.simple.SimpleMatrix;

public class MatrixPanel extends JPanel {

	private SimpleMatrix dataMatrix;
	
	public MatrixPanel(SimpleMatrix dataMatrix){
		this.dataMatrix = dataMatrix;
	}
	
	public void updateData(SimpleMatrix m){
		dataMatrix = m;
	}
	
	@Override
	protected void paintComponent(Graphics g) {
		super.paintComponent(g);
		
		// Clear the map
		g.clearRect(0, 0, getWidth(), getHeight());

		// Draw the grid
		int cellWidth = getWidth() / dataMatrix.numCols();
		int cellHeight = getHeight() / dataMatrix.numRows();
		
		for (int col = 0; col < dataMatrix.numCols(); col++) {
			for (int row = 0; row <dataMatrix.numRows(); row++) {
				double value = dataMatrix.get(row, col);
				int rgb = (int) ((1-value) * 255);
				Color c = new Color(rgb, rgb, rgb);
				g.setColor(c);
				g.fillRect((int)(col*cellWidth), (int)(row*cellHeight),
							(int)cellWidth, (int)cellHeight);
			}
		}

	}
}
