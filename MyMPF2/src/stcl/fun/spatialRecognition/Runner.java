package stcl.fun.spatialRecognition;

import java.awt.Dimension;
import java.util.Random;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SpatialPooler;
import stcl.algo.som.SomNode;
import stcl.graphics.MapDrawer;
import stcl.graphics.MapDrawerBW;
import stcl.graphics.MapDrawerGRAY;

public class Runner {
	
	private MapDrawer frame;
	private SpatialPooler pooler;
	SimpleMatrix bigT;
	
	
	public static void main(String[] args) {
		Runner runner = new Runner();
		runner.run();
	}
	
	public void run(){
		int FRAMES_PER_SECOND = 1;
	    int SKIP_TICKS = 1000 / FRAMES_PER_SECOND;
	   
	    float next_game_tick = System.currentTimeMillis();
	    float sleepTime = 0;
		
		setupExperiment();
		
		int maxIterations = 10;
		
		for (int i = 0; i < maxIterations; i++){
			//Feed forward
			SimpleMatrix out = pooler.feedForward(bigT);			
			
			//Collect BMU
			SomNode bmu = pooler.getSOM().getBMU(bigT);
			
			//Visualize bmu
			SimpleMatrix m = new SimpleMatrix(bmu.getVector());
			m.reshape(5, 5);
			updateGraphics(m);
			next_game_tick+= SKIP_TICKS;
			sleepTime = next_game_tick - System.currentTimeMillis();
			if (sleepTime >= 0){
				try {
					Thread.sleep(SKIP_TICKS);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			pooler.tick();
		}
				
			
	}
	
	private void setupExperiment(){
		//Create Figure matrices
		//SimpleMatrix[] matrices = matrices();
		
		bigT = bigT();
		
		//Create spatial pooler
		Random rand = new Random();
		int maxIterations = 10;
		int inputLength = 5*5;
		int mapSize = 10;
		pooler = new SpatialPooler(rand, maxIterations, inputLength, mapSize);
		
		//Setup graphics
		setupGraphics(5, 5);
		
	}
	
	private void updateGraphics(SimpleMatrix matrix){
		frame.updateMap(matrix);
		frame.revalidate();
		frame.repaint();
	}
	
	private void setupGraphics(int mapHeight, int mapWidth ) {
		frame = new MapDrawerGRAY(mapHeight, mapWidth);
		frame.setSize(new Dimension(400, 400));
		frame.setTitle("Visualiztion");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.pack();
		SimpleMatrix m = new SimpleMatrix(5, 5);
		frame.updateMap(m);
		frame.setVisible(true);

	}
	
	private SimpleMatrix[] matrices(){
		SimpleMatrix[] matrices = new SimpleMatrix[4];
		double[][] bigTData = {
				{0,0,0,0,0},
				{0,1,1,1,0},
				{0,0,1,0,0},
				{0,0,1,0,0},
				{0,0,1,0,0}};
		SimpleMatrix bigT = new SimpleMatrix(bigTData);
		matrices[0] = bigT;
		
		double[][] smallOData = {
				{0,0,0,0,0},
				{0,1,1,1,0},
				{0,1,0,1,0},
				{0,1,1,1,0},
				{0,0,0,0,0}};
		SimpleMatrix smallO = new SimpleMatrix(smallOData);
		matrices[1] = smallO;
		
		double[][] bigOData = {
				{1,1,1,1,1},
				{1,0,0,0,1},
				{1,0,0,0,1},
				{1,0,0,0,1},
				{1,1,1,1,1}};
		SimpleMatrix bigO = new SimpleMatrix(bigOData);
		matrices[2] = bigO;
		
		double[][] smallVData = {
				{0,0,0,0,0},
				{0,0,0,0,0},
				{1,0,0,0,1},
				{0,1,0,1,0},
				{0,0,1,0,0}};
		SimpleMatrix smallV = new SimpleMatrix(smallVData);
		matrices[3] = smallV;
		
		return matrices;
	}
	
	private SimpleMatrix bigT(){
		double[][] bigTData = {
				{0,0,0,0,0},
				{0,1,1,1,0},
				{0,0,1,0,0},
				{0,0,1,0,0},
				{0,0,1,0,0}};
		SimpleMatrix bigT = new SimpleMatrix(bigTData);
		bigT.reshape(1, bigT.numCols() * bigT.numRows());
		return bigT;
	}

	
	
	
	

}
