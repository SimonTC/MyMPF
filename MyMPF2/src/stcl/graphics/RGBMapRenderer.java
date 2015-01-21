/*
 * LatticeRenderer.java
 *
 * Created on December 13, 2002, 2:55 PM
 */

package stcl.graphics;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.som.som.SOM;

/**
 *
 * @author  alanter
 */
public class RGBMapRenderer extends MapRenderer {
	private BufferedImage img = null;
	Font arialFont = new Font("Arial", Font.BOLD, 12);
	SOM lattice;
	boolean ready = false;
	
	/** Creates a new instance of LatticeRenderer */
	public RGBMapRenderer() {
		lattice = null;
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
		lattice = lat;
		ready = true;
	}
	
	// Yeah, it's ugly.  But it works, and I didn't feel like commenting it. :)
	// All it does it slaps the given lattice's weight values up in a 2x2
	// grid as an image
	public void render(SOM lattice, int iteration) {
		float cellWidth = (float)getWidth() / (float)lattice.getWidth();
		float cellHeight = (float)getHeight() / (float)lattice.getHeight();
//		float cellWidth = 2;
//		float cellHeight = 2;
		
		int imgW = img.getWidth();
		int imgH = img.getHeight();
		float r, g, b;
		Graphics2D g2 = img.createGraphics();
		g2.setBackground(Color.black);
		g2.clearRect(0,0,imgW,imgH);
		for (int x=0; x<lattice.getWidth(); x++) {
			for (int y=0; y<lattice.getHeight(); y++) {

				r = (float)(lattice.getModel(y, x).getVector().get(0));
				g = (float)(lattice.getModel(y, x).getVector().get(1));
				b = (float)(lattice.getModel(y, x).getVector().get(2));
				g2.setColor(new Color(r,g,b));
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
		if (!ready) return;
		int x = evt.getX();
		int y = evt.getY();
		float cellWidth = (float)getWidth() / (float)lattice.getWidth();
		float cellHeight = (float)getHeight() / (float)lattice.getHeight();
		int cellX = (int)((float)x / cellWidth);
		int cellY = (int)((float)y / cellHeight);
		Graphics2D g2 = (Graphics2D)img.getGraphics();
		g2.setFont(arialFont);
		float r, g , b;
		r = (float)(lattice.getModel(cellY, cellX).getVector().get(0));
		g = (float)(lattice.getModel(cellY, cellX).getVector().get(1));
		b = (float)(lattice.getModel(cellY, cellX).getVector().get(2));
		StringBuffer sb = new StringBuffer("Value: ");
		sb.append(String.valueOf(r)).append(", ");
		sb.append(String.valueOf(g)).append(", ");
		sb.append(String.valueOf(b));
		g2.setColor(Color.black);
		g2.fillRect(0, getHeight() - 22, getWidth(), 15);
		g2.setColor(Color.white);
		g2.drawString(sb.toString(), 5, getHeight() - 10);
		g2.dispose();
		repaint();
	}

	@Override
	public void render(SimpleMatrix matrix, int iteration) {
		// TODO Auto-generated method stub
		
	}
}