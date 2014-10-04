package stcl.graphics;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;

import javax.swing.JFrame;
import javax.swing.JPanel;

import org.ejml.simple.SimpleMatrix;

public class MultipleMapDrawerGRAY extends JFrame {

	// Info about the map
	SimpleMatrix[] dataMaps;

	private int rowsInMap;
	private int columnsInMap;
	private MapPanel[] panels;

	public MultipleMapDrawerGRAY(int rowsInMap, int columnsInMap, int numberOfMaps, int prefMapWidth, int prefMapHeight) {
		// Read values
		this.columnsInMap = columnsInMap;
		this.rowsInMap = rowsInMap;
		
		int spaceBetweenMaps = 20;

		// Set size of map
		setPreferredSize(new Dimension((prefMapWidth + spaceBetweenMaps) * numberOfMaps, (prefMapHeight + spaceBetweenMaps) * numberOfMaps));
		
		// Add MapPanels
		panels = new MapPanel[numberOfMaps];
		for ( int i = 0; i < numberOfMaps; i++){
			MapPanel panel = new MapPanel(i);
			panels[i] = panel;
			add(panel);
		}
		
	}

	/**
	 * Updates the colorGrid Has to be called before .repaint() Else no effect
	 * on the map
	 * 
	 * @param newColorGrid
	 */
	public void updateMaps(SimpleMatrix[] dataMaps) {
		this.dataMaps = dataMaps;
		for (MapPanel p: panels){
			p.revalidate();
			p.repaint();
		}
	}

	class MapPanel extends JPanel {
		private int id;
		public MapPanel(int id){
			super();
			this.id = id;
		}
		@Override
		protected void paintComponent(Graphics g) {
			super.paintComponent(g);

			// Clear the map
			g.clearRect(0, 0, getWidth(), getHeight());

			// Draw the grid
				int cellWidth = getWidth() / columnsInMap;
			int cellHeight = getHeight() / rowsInMap;
			
			for (int x=0; x<columnsInMap; x++) {
				for (int y=0; y<rowsInMap; y++) {
					double value = dataMaps[id].get(y, x);
					int rgb = (int) ((1-value) * 255);
					Color c = new Color(rgb, rgb, rgb);
					g.setColor(c);
					g.fillRect((int)(x*cellWidth), (int)(y*cellHeight),
								(int)cellWidth, (int)cellHeight);
				}
			}
		}
	}

}
