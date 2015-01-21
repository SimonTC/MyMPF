package stcl.graphics;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;

import javax.swing.JPanel;

import org.ejml.simple.SimpleMatrix;

public class MapDrawerBW extends MapDrawer {

	// Info about the map
	//public static final int PREFERRED_GRID_SIZE_PIXELS = 10;

	SimpleMatrix dataMap;

	private int mapHeight;
	private int mapWidth;
	private MapPanel panel;

	public MapDrawerBW(int mapHeight, int mapWidth) {
		// Read values
		this.mapWidth = mapWidth;
		this.mapHeight = mapHeight;

		// Set size of map
		setPreferredSize(new Dimension(400, 400));
		
		// Add MapPanel
		panel = new MapPanel();
		add(panel);
	}

	/**
	 * Updates the colorGrid Has to be called before .repaint() Else no effect
	 * on the map
	 * 
	 * @param newColorGrid
	 */
	public void updateMap(SimpleMatrix dataMap) {
		this.dataMap = dataMap;
		this.panel.revalidate();
		this.panel.repaint();
	}

	class MapPanel extends JPanel {
		@Override
		protected void paintComponent(Graphics g) {
			super.paintComponent(g);

			// Clear the map
			g.clearRect(0, 0, getWidth(), getHeight());

			// Draw the grid
			int cellWidth = getWidth() / mapWidth;
			int cellHeight = getHeight() / mapHeight;
			
			for (int x=0; x<mapWidth; x++) {
				for (int y=0; y<mapHeight; y++) {
					double value = dataMap.get(y, x);
					Color c;
					if (value > 0.5){
						c = Color.black;
					} else{
						c = Color.white;
					}
					g.setColor(c);
					g.fillRect((int)(x*cellWidth), (int)(y*cellHeight),
								(int)cellWidth, (int)cellHeight);
				}
			}
		}
	}

}