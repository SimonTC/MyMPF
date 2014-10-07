package stcl.fun.movingLines;

import java.util.Random;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.algo.som.SOM;
import stcl.algo.som.SomNode;

public class Runner {
	private SimpleMatrix[][] sequences;
	private SpatialPooler spatialPooler;
	private TemporalPooler temporalPooler;
	private MovingLinesGUI frame;
	private SOM possibleInputs;
	
	public static void main(String[] args){
		Runner runner = new Runner();
		runner.run();
	}
	
	public void run(){
		//Setup experiment
		int maxIterations = 1000;
		setupExperiment(maxIterations);
		
		//Setup graphics
		setupGraphics();
		
		runExperiment(maxIterations);
		
	}
	
	private void runExperiment(int maxIterations){
		int FRAMES_PER_SECOND = 5;
	    int SKIP_TICKS = 1000 / FRAMES_PER_SECOND;
	   
	    float next_game_tick = System.currentTimeMillis();
	    float sleepTime = 0;
	    int timeSinceNotBlank = 0;
	    Random rand = new Random();
	    
	    for (int i = 0; i < maxIterations; i++){
	    	
	    	//Choose sequence
	    	//SimpleMatrix[] curSequence;
	    /*
	    	int notBlankThreshold = 30 * timeSinceNotBlank;
	    	int value = rand.nextInt(100);
	    	if (value < notBlankThreshold){
	    		if (rand.nextBoolean()){
	    			curSequence = sequences[0]; //Horizontal down
	    		} else {
	    			curSequence = sequences[1]; //Vertical right
	    		}
	    		timeSinceNotBlank = 0;
	    	} else {
	    		curSequence = sequences[2]; //Blank
	    		timeSinceNotBlank++;
	    	}
	    	
	    	*/
	    	//Got through all sequences
	    	for (SimpleMatrix[] curSequence : sequences){
		    	//Go through sequence
		    	for (int j = 0; j < curSequence.length; j++){
		    		//Spatial classification
		    		SimpleMatrix spatialFFOutputMatrix = spatialPooler.feedForward(curSequence[j]);
		    		
		    		//Normalize output
		    		double sum = spatialFFOutputMatrix.elementSum();
		    		spatialFFOutputMatrix = spatialFFOutputMatrix.scale(1/sum);
		    		
		    		//Transform spatial output matrix to vector
		    		double[] spatialFFOutputDataVector = spatialFFOutputMatrix.getMatrix().data;		
		    		SimpleMatrix temporalFFInputVector = new SimpleMatrix(1, spatialFFOutputDataVector.length);
		    		temporalFFInputVector.getMatrix().data = spatialFFOutputDataVector;
		    		
		    		//Orthogonalize spatial output
		    		temporalFFInputVector = orthogonalize(temporalFFInputVector);
		    		
		    		//Temporal classification
		    		temporalPooler.feedForward(temporalFFInputVector);
		    		
		    		//Update graphics
		    		updateGraphics(curSequence[j], i);
		    		
		    		//Sleep
					next_game_tick+= SKIP_TICKS;
					sleepTime = next_game_tick - System.currentTimeMillis();
					try {
						Thread.sleep(SKIP_TICKS);
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					
		    		
		    	}	
		    	temporalPooler.resetLeakyDifferences();
	    	}
			
			spatialPooler.tick();
			temporalPooler.tick();
			

	    }
	}
	
	
	private SimpleMatrix orthogonalize(SimpleMatrix m){
		double maxValue = 0;
		int maxID = -1;
		for (int row = 0; row < m.numRows(); row++){
			for (int col = 0; col < m.numCols(); col++){
				double value = m.get(row, col);
				if (value > maxValue){
					maxValue = value;
					maxID = m.getIndex(row, col);
				}
			}
		}
		
		m.set(0);
		m.set(maxID, maxValue);
		return m;
	}
	
	private void setupGraphics(){
		frame = new MovingLinesGUI(spatialPooler.getSOM(), possibleInputs);
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
	
	private void setupExperiment(int maxIterations){
		buildSequences();
		buildPoolers(maxIterations);		
	}
	
	private void buildPoolers(int maxIterations){
		Random rand = new Random();
		
		//Spatial pooler
		int spatialInputLength = 9;
		int spatialMapSize = 5;
		double initialLearningRate = 0.8;
		double initialNeighborhoodRadius = 3; //3
		spatialPooler = new SpatialPooler(rand, maxIterations, spatialInputLength, spatialMapSize, initialLearningRate,initialNeighborhoodRadius,1);
		
		//Temporal pooler
		int temporalInputLength = spatialMapSize * spatialMapSize;
		int temporalMapSize = 2;
		double initialTemporalLeakyCoefficient = 0.6;
		temporalPooler = new TemporalPooler(rand, maxIterations, temporalInputLength, temporalMapSize, initialTemporalLeakyCoefficient);
	}
	
	private void buildSequences(){
		sequences = new SimpleMatrix[3][3];
		possibleInputs = new SOM(3, 3, 9, new Random());
		SomNode[] nodes = possibleInputs.getModels();
		
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
	}
}
