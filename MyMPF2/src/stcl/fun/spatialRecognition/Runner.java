package stcl.fun.spatialRecognition;

import java.util.Random;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SpatialPooler;
import stcl.algo.som.SomNode;
import stcl.graphics.MultipleMapDrawerGRAY;

public class Runner {
	
	private MultipleMapDrawerGRAY frame;
	private SpatialPooler pooler;
	private SimpleMatrix[] figureMatrices;
	
	public static void main(String[] args) {
		Runner runner = new Runner();
		runner.run();
	}
	
	public void run(){
		int FRAMES_PER_SECOND = 2;
	    int SKIP_TICKS = 1000 / FRAMES_PER_SECOND;
	   
	    float next_game_tick = System.currentTimeMillis();
	    float sleepTime = 0;
		
	    int maxIterations = 50;
	    
		setupExperiment(maxIterations, true);
		
		
		
		for (int i = 0; i < maxIterations; i++){
			SimpleMatrix[] outputs = new SimpleMatrix[figureMatrices.length];
			
			for (int j = 0; j < figureMatrices.length; j++){
				//Feed forward
				SimpleMatrix out = pooler.feedForward(figureMatrices[j]);			
				
				//Collect BMU
				SomNode bmu = pooler.getSOM().getBMU(figureMatrices[j]);
				SimpleMatrix m = new SimpleMatrix(bmu.getVector());
				m.reshape(5, 5);
				outputs[j] = m;				
			}
			
			
			//Visualize maps
			updateGraphics(outputs, i);
			
			//Sleep
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
	
	private void setupExperiment(int iterations, boolean simple){
		int figureRows = 0;
		int figureColumns = 0;
		
		//Create Figure matrices
		if (simple){
			figureMatrices = simpleFigures();
			figureRows = 5;
			figureColumns = 5;
		}

		
		//Create spatial pooler
		Random rand = new Random();
		int maxIterations = iterations;
		int inputLength = figureColumns * figureRows;
		int mapSize = 10;
		pooler = new SpatialPooler(rand, maxIterations, inputLength, mapSize, 0.2, 5, 1);
		
		//Setup graphics
		setupGraphics(figureRows, figureColumns);
		
	}
	
	private void updateGraphics(SimpleMatrix[] matrices, int iteration){
		frame.updateMaps(matrices);
		frame.setTitle("Visualiztion - Iteration: " + iteration);
		frame.revalidate();
		frame.repaint();
	}
	
	private void setupGraphics(int mapHeight, int mapWidth ) {
		int mapGuiSize = 200;
		frame = new MultipleMapDrawerGRAY(mapHeight, mapWidth, figureMatrices.length, mapGuiSize, mapGuiSize);
		frame.setTitle("Visualiztion");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		SimpleMatrix[] tmp = new SimpleMatrix[4];
		for (int i = 0; i < 4; i++){
			SimpleMatrix m = new SimpleMatrix(mapHeight, mapWidth);
			tmp[i] = m;
		}
		
		frame.updateMaps(tmp);
		frame.pack();

		frame.setVisible(true);

	}
	
	private SimpleMatrix[] simpleFigures(){
		SimpleMatrix[] matrices = new SimpleMatrix[4];
		
		double[][] bigTData = {
				{0,0,0,0,0},
				{0,1,1,1,0},
				{0,0,1,0,0},
				{0,0,1,0,0},
				{0,0,1,0,0}};
		SimpleMatrix bigT = new SimpleMatrix(bigTData);
		bigT.reshape(1, bigT.numCols() * bigT.numRows());
		matrices[0] = bigT;
		
		double[][] smallOData = {
				{0,0,0,0,0},
				{0,1,1,1,0},
				{0,1,0,1,0},
				{0,1,1,1,0},
				{0,0,0,0,0}};
		SimpleMatrix smallO = new SimpleMatrix(smallOData);
		smallO.reshape(1, smallO.numCols() * smallO.numRows());
		matrices[1] = smallO;
		
		double[][] bigOData = {
				{1,1,1,1,1},
				{1,0,0,0,1},
				{1,0,0,0,1},
				{1,0,0,0,1},
				{1,1,1,1,1}};
		SimpleMatrix bigO = new SimpleMatrix(bigOData);
		bigO.reshape(1, bigO.numCols() * bigO.numRows());
		matrices[2] = bigO;
		
		double[][] smallVData = {
				{0,0,0,0,0},
				{0,0,0,0,0},
				{1,0,0,0,1},
				{0,1,0,1,0},
				{0,0,1,0,0}};
		SimpleMatrix smallV = new SimpleMatrix(smallVData);
		smallV.reshape(1, smallV.numCols() * smallV.numRows());
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
