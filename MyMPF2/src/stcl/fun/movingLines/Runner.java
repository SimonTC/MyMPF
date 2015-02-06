package stcl.fun.movingLines;

import java.util.Random;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.graphics.MovingLinesGUI_Prediction;
import dk.stcl.som.containers.SomNode;
import dk.stcl.som.som.SOM;

public class Runner {
	private SimpleMatrix[][] sequences;
	private SpatialPooler spatialPooler;
	private TemporalPooler temporalPooler;
	private MovingLinesGUI_Prediction frame;
	private SOM possibleInputs;
	private Random rand = new Random(1234);
	
	private final int ITERATIONS = 10000;
	private final boolean VISUALIZE_TRAINING = false;
	private final boolean VISUALIZE_RESULT = true;
	
	public static void main(String[] args){
		Runner runner = new Runner();
		runner.run();
	}
	
	public void run(){
		//Setup experiment		
		setupExperiment(ITERATIONS, rand);
		
		//Setup graphics
		if (VISUALIZE_TRAINING) setupGraphics();
		
		runExperiment(ITERATIONS, rand, VISUALIZE_TRAINING);
		
		if (VISUALIZE_RESULT){
			temporalPooler.flushTemporalMemory();
			temporalPooler.setLearning(false);
			 setupGraphics();
			 runExperiment(ITERATIONS, rand, true);
			 
		}
		
	}
	
	private void runExperiment(int maxIterations, Random rand, boolean visualize){
		int FRAMES_PER_SECOND = 20;
	    int SKIP_TICKS = 1000 / FRAMES_PER_SECOND;
	   
	    float next_game_tick = System.currentTimeMillis();
	    int curSeqID = 0;
    	int curInputID = -1;
	    
	    for (int i = 0; i < maxIterations; i++){
	    	//Choose sequence
	    	
	    	boolean change = rand.nextDouble() > 0.9 ? true : false;
	    	SimpleMatrix[] curSequence = null;
			if (change){
				//temporalPooler.flushTemporalMemory();
				boolean choose = rand.nextBoolean();
				switch (curSeqID){
				case 0 : curSeqID = choose ? 1 : 2; break;
				case 1 : curSeqID = choose ? 2 : 0; break;
				case 2 : curSeqID = choose ? 0 : 1; break;
				}
				curInputID = -1;
			} 
			curSequence = sequences[curSeqID];
			curInputID++;
			curInputID = curInputID >= curSequence.length? 0 : curInputID;
	    
    		//Spatial classification
    		SimpleMatrix spatialFFOutputMatrix = spatialPooler.feedForward(curSequence[curInputID]);
    		
    		/*
    		//Normalize output
    		double max = spatialFFOutputMatrix.elementSum();
    		SimpleMatrix temporalFFInputVector = spatialFFOutputMatrix.scale(1/max);
    		*/
    		    		
    		SimpleMatrix temporalFFInputVector = new SimpleMatrix(spatialFFOutputMatrix);
    		
    		//Transform spatial output matrix to vector
    		temporalFFInputVector.reshape(1, spatialFFOutputMatrix.getMatrix().data.length);
    		
    		//Temporal classification
    		temporalPooler.feedForward(temporalFFInputVector);
    		
    		if (visualize){
	    		//Update graphics
	    		updateGraphics(curSequence[curInputID], i);
	    		
	    		//Sleep
				next_game_tick+= SKIP_TICKS;
				try {
					Thread.sleep(SKIP_TICKS);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}	
    		}
	    }
	}
	
	private void setupGraphics(){
		frame = new MovingLinesGUI_Prediction(spatialPooler, temporalPooler);
		frame.setTitle("Visualiztion");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		updateGraphics(sequences[2][0],0); //Give a blank
		frame.pack();
		frame.setVisible(true);
		
	}
	
	private void updateGraphics(SimpleMatrix inputVector, int iteration){
		frame.updateData(inputVector, spatialPooler, temporalPooler);
		frame.setTitle("Visualiztion - Iteration: " + iteration);
		frame.revalidate();
		frame.repaint();

	}
	
	private void setupExperiment(int maxIterations, Random rand){
		buildSequences();
		buildPoolers(maxIterations, rand);		
	}
	
	private void buildPoolers(int maxIterations, Random rand){
		
		//Spatial pooler
		int spatialInputLength = 9;
		int spatialMapSize = 3;
		double initialLearningRate = 0.1;
		spatialPooler = new SpatialPooler(rand, spatialInputLength, spatialMapSize, initialLearningRate,2,0.125);
		
		//Temporal pooler
		int temporalInputLength = spatialMapSize * spatialMapSize;
		int temporalMapSize = 2;
		double initialTemporalLeakyCoefficient = 0.3;
		double stdDev = 2;
		temporalPooler = new TemporalPooler(rand, temporalInputLength, temporalMapSize, 0.1, stdDev, 0.125, initialTemporalLeakyCoefficient);
	}
	
	private void buildSequences(){
		sequences = new SimpleMatrix[4][3];
		possibleInputs = new SOM(3, 4, 9, new Random(), 0.1, 1, 0.125);
		SomNode[] nodes = possibleInputs.getNodes();
		
		SimpleMatrix m;
		
		//Horizontal down
		double[][] hor1 = {
				{1,1,1},
				{0,0,0},
				{0,0,0}};
		m = new SimpleMatrix(hor1);
		m.reshape(1, 9);
		sequences[0][0] = m;
		nodes[0] = new SomNode(m);
		
		double[][] hor2 = {
				{0,0,0},
				{1,1,1},
				{0,0,0}};
		m = new SimpleMatrix(hor2);
		m.reshape(1, 9);
		sequences[0][1] = m;
		nodes[1] = new SomNode(m);
		
		double[][] hor3 = {
				{0,0,0},
				{0,0,0},
				{1,1,1}};
		m = new SimpleMatrix(hor3);
		m.reshape(1, 9);
		sequences[0][2] = m;
		nodes[2] = new SomNode(m);
		
		//Vertical right
		double[][] ver1 = {
				{1,0,0},
				{1,0,0},
				{1,0,0}};
		m = new SimpleMatrix(ver1);
		m.reshape(1, 9);
		sequences[1][0] = m;
		nodes[3] = new SomNode(m);
		
		double[][] ver2 = {
				{0,1,0},
				{0,1,0},
				{0,1,0}};
		m = new SimpleMatrix(ver2);
		m.reshape(1, 9);
		sequences[1][1] = m;
		nodes[4] = new SomNode(m);
		
		double[][] ver3 = {
				{0,0,1},
				{0,0,1},
				{0,0,1}};
		m = new SimpleMatrix(ver3);
		m.reshape(1, 9);
		sequences[1][2] = m;
		nodes[5] = new SomNode(m);
		
		//Blank
		double[][] blank = {
				{0,0,0},
				{0,0,0},
				{0,0,0}};
		m = new SimpleMatrix(blank);
		m.reshape(1, 9);
		sequences[2][0] = m;
		sequences[2][1] = m;
		sequences[2][2] = m;
		nodes[6] = new SomNode(m);
		nodes[7] = new SomNode(m);
		nodes[8] = new SomNode(m);
		
		//Vertical left
		m = new SimpleMatrix(ver3);
		m.reshape(1, 9);
		sequences[3][0] = m;
		nodes[9] = new SomNode(m);
		
		m = new SimpleMatrix(ver2);
		m.reshape(1, 9);
		sequences[3][1] = m;
		nodes[10] = new SomNode(m);
		
		m = new SimpleMatrix(ver1);
		m.reshape(1, 9);
		sequences[3][2] = m;
		nodes[11] = new SomNode(m);
		
	}
}
