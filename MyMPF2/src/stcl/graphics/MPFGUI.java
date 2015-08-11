package stcl.graphics;

import java.awt.Dimension;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.Network;

public class MPFGUI {
	
	private JFrame frame;
	private NetworkPanel overview;
	private int pause;
	private String curSequenceName;
	
	public void initialize(int observationMatrixSize, int actionMatrixSize, int fps){
		frame = new JFrame();
		overview = new NetworkPanel();
		frame.setPreferredSize(new Dimension(1200, 500));
		frame.setTitle("Visualization");
		overview.initialize(observationMatrixSize, actionMatrixSize);
		frame.getContentPane().add(overview);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.pack();
		frame.setVisible(true);	
		pause = 1000 / fps;
	}

	
	public void update(Network brain, SimpleMatrix inputNow, SimpleMatrix actionNow, SimpleMatrix prediction, SimpleMatrix actionNext, int step){
		overview.updateData(inputNow, actionNow, prediction, actionNext);
		frame.setTitle("Sequence " + curSequenceName + " Step " + step);
		frame.revalidate();
		frame.repaint();
		
		try {
			Thread.sleep(pause);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void setSequenceName(String name){
		this.curSequenceName = name;
	}
}
