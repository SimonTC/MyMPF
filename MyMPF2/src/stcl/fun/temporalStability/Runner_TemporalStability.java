package stcl.fun.temporalStability;

import java.util.ArrayList;
import java.util.Random;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.fun.movingLines.MovingLinesGUI;
import dk.stcl.som.containers.SomNode;
import dk.stcl.som.som.SOM;

public class Runner_TemporalStability {
	private ArrayList<SimpleMatrix[]> sequences;
	private SpatialPooler spatialPooler;
	private TemporalPooler temporalPooler;
	private MovingLinesGUI frame;
	private SOM possibleInputs;
	private Random rand = new Random(1234);
	
	private final int ITERATIONS = 10000;
	private final boolean VISUALIZE_TRAINING = false;
	private final boolean VISUALIZE_RESULT = true;
	private SimpleMatrix bigT;
	private SimpleMatrix smallO;
	private SimpleMatrix bigO;
	private SimpleMatrix smallV;
	private SimpleMatrix blank;
	
	public static void main(String[] args){
		Runner_TemporalStability runner = new Runner_TemporalStability();
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
				int nextSeqID;
				do {
					nextSeqID = rand.nextInt(sequences.size());
				} while (nextSeqID == curSeqID);
				
				curSeqID = nextSeqID;
				curInputID = -1;
			} 
			curSequence = sequences.get(curSeqID);
			curInputID++;
			curInputID = curInputID >= curSequence.length? 0 : curInputID;
	    
    		//Spatial classification
    		SimpleMatrix spatialFFOutputMatrix = spatialPooler.feedForward(curSequence[curInputID]);
    		
    		//Transform spatial output matrix to vector
    		SimpleMatrix temporalFFInputVector = new SimpleMatrix(spatialFFOutputMatrix);
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
		frame = new MovingLinesGUI(spatialPooler.getSOM(), possibleInputs);
		frame.setTitle("Visualiztion");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		updateGraphics(blank,0); //Give a blank
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
		int spatialInputLength = blank.getNumElements();
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
		createFigures();
		sequences = new ArrayList<SimpleMatrix[]>();
		possibleInputs = new SOM(3, 4, 9, new Random(), 0.1, 1, 0.125);

		SimpleMatrix[] small_1 = {bigT};
		SimpleMatrix[] small_2 = {bigO};
		SimpleMatrix[] small_3 = {smallO};
		
		sequences.add(small_1);
		sequences.add(small_2);
		sequences.add(small_3);
		
		
	}
	
	private void createFigures(){		
		double[][] bigTData = {
				{0,0,0,0,0},
				{0,1,1,1,0},
				{0,0,1,0,0},
				{0,0,1,0,0},
				{0,0,1,0,0}};
		bigT = new SimpleMatrix(bigTData);
		bigT.reshape(1, bigT.numCols() * bigT.numRows());
		
		double[][] smallOData = {
				{0,0,0,0,0},
				{0,1,1,1,0},
				{0,1,0,1,0},
				{0,1,1,1,0},
				{0,0,0,0,0}};
		smallO = new SimpleMatrix(smallOData);
		smallO.reshape(1, smallO.numCols() * smallO.numRows());
		
		double[][] bigOData = {
				{1,1,1,1,1},
				{1,0,0,0,1},
				{1,0,0,0,1},
				{1,0,0,0,1},
				{1,1,1,1,1}};
		bigO = new SimpleMatrix(bigOData);
		bigO.reshape(1, bigO.numCols() * bigO.numRows());
		
		double[][] smallVData = {
				{0,0,0,0,0},
				{0,0,0,0,0},
				{1,0,0,0,1},
				{0,1,0,1,0},
				{0,0,1,0,0}};
		smallV = new SimpleMatrix(smallVData);
		smallV.reshape(1, smallV.numCols() * smallV.numRows());
		
		double[][] blankData = {
				{0,0,0,0,0},
				{0,0,0,0,0},
				{0,0,0,0,0},
				{0,0,0,0,0},
				{0,0,0,0,0}};
		blank = new SimpleMatrix(blankData);
		blank.reshape(1, blank.numCols() * blank.numRows());
	}
}
