/*
 * LatticeRenderer.java
 *
 * Created on December 13, 2002, 2:55 PM
 */

package stcl.graphics;

import javax.swing.JPanel;

import stcl.algo.som.SOM;

import java.awt.image.BufferedImage;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;

/**
 *
 * @author  alanter
 */
public class BWMapRenderer extends JPanel {
	private BufferedImage img = null;
	Font arialFont = new Font("Arial", Font.BOLD, 12);
	SOM map;
	boolean ready = false;
	
	
	public BWMapRenderer() {
		map = null;
        addMouseMotionListener(new java.awt.event.MouseMotionAdapter() {
            public void mouseMoved(java.awt.event.MouseEvent evt) {
                panelMouseMoved(evt);
            }
        });
	}

	public void paint(Graphics g) {
		if (img == null)
			super.paint(g);
		else
			g.drawImage(img, 0, 0, this);
	}
	
	public void registerLattice(SOM lat) {
		map = lat;
		ready = true;
	}
	
	public void render(SOM map, int iteration) {
		float cellWidth = (float)getWidth() / (float)map.getWidth();
		float cellHeight = (float)getHeight() / (float)map.getHeight();
		
		int imgW = img.getWidth();
		int imgH = img.getHeight();
		double black;
		Graphics2D g2 = img.createGraphics();
		g2.setBackground(Color.white);
		g2.clearRect(0,0,imgW,imgH);
		for (int x=0; x<map.getWidth(); x++) {
			for (int y=0; y<map.getHeight(); y++) {
				black = map.getModel(y, x).getVector().get(0);
				Color c;
				if (black > 0.5){
					c = Color.black;
				} else{
					c = Color.white;
				}
				g2.setColor(c);
				g2.fillRect((int)(x*cellWidth), (int)(y*cellHeight),
							(int)cellWidth, (int)cellHeight);
			}
		}
		g2.setColor(Color.black);
		g2.setFont(arialFont);
		g2.drawString("Iteration: " + String.valueOf(iteration), 5, 15);
		g2.dispose();
		repaint();
	}
	
	public BufferedImage getImage() {
		if (img == null)
			img = (BufferedImage)createImage(getWidth(), getHeight());
		
		return img;
	}
	
	public void setImage(BufferedImage bimg) {
		img = bimg;
	}
	
	private void panelMouseMoved(java.awt.event.MouseEvent evt) {
		//Nothing happens on mouse over
	}
}
