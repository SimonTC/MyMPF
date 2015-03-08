package stcl.fun.temporalRecognition;

import java.util.ArrayList;
import java.util.Random;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import stcl.graphics.MovingLinesGUI_Prediction;
import dk.stcl.core.rsom.IRSOM;
import dk.stcl.core.rsom.RSOM_SemiOnline;
import dk.stcl.core.rsom.RSOM_Simple;
import dk.stcl.core.som.ISOM;
import dk.stcl.core.som.SOM_SemiOnline;

public class TemporalRecognition_RSOM {
	
	private IRSOM rsom;
	private ArrayList<SimpleMatrix[]> sequences;
	private SimpleMatrix bigT, bigO, smallO, smallV, blank;
	private MovingLinesGUI_Prediction frame;
	private Random rand = new Random(1234);
	private SOM_SemiOnline spatialDummy = new SOM_SemiOnline(2,5,rand,0,0,0);
	
	
	
	private final int ITERATIONS = 10000;
	private final boolean VISUALIZE_TRAINING = false;
	private final boolean VISUALIZE_RESULT = true;
	private final int GUI_SIZE = 500;
	private final int FRAMES_PER_SECOND = 10;
	private final int MAP_SIZE = 2;
	private final boolean USE_SIMPLE_RSOM = false;

	public static void main(String[] args){
		TemporalRecognition_RSOM runner = new TemporalRecognition_RSOM();
		runner.run();
	}

	public void run(){
		//Setup experiment		
		setupExperiment(ITERATIONS, rand);
		
		//Setup graphics
		if (VISUALIZE_TRAINING) setupVisualization(spatialDummy, GUI_SIZE);
		
		//Train
		training(ITERATIONS, rand, VISUALIZE_TRAINING);
		
		//Label
		int[] labels = createLabels();
		RSomLabeler labeler = new RSomLabeler();
		labeler.label(rsom, sequences, labels);
		
		//Evaluate
		
		//Print info
		rsom.printLabelMap();
		/*
	    System.out.println("Rsom models:");
	    System.out.println();
	    for (SomNode n : rsom.getNodes()){
	    	SimpleMatrix vector = new SimpleMatrix(n.getVector());
	    	vector.reshape(2, 2);
	    	vector.print();
	    	System.out.println(vector.elementSum());
	    	System.out.println();
	    }
		 */
	    
		if (VISUALIZE_RESULT){
			rsom.flush();
			rsom.setLearning(false);
			setupVisualization(spatialDummy, GUI_SIZE);
			training(ITERATIONS, rand, true);
			 
		}
		
	}
	private int[] createLabels(){
		int[] labels = new int[sequences.size()];
		for (int i = 0; i < labels.length; i++){
			labels[i] = i;
		}
		
		return labels;
	}

	private void setupExperiment(int maxIterations, Random rand){
		buildSequences();
		buildRSOM();
	}
	
	private void buildRSOM(){
		int mapSize = MAP_SIZE;
		int inputLength = blank.getNumElements();
		double learningRate = 0.1;
		double stddev = 1;
		double activationCodingFactor = 0.125;
		double decay = 0.3;
		
		if (USE_SIMPLE_RSOM){
			rsom = new RSOM_Simple(mapSize, inputLength, rand, learningRate, activationCodingFactor, ITERATIONS, decay);
		} else {
			rsom = new RSOM_SemiOnline(mapSize, inputLength, rand, learningRate, activationCodingFactor, stddev, decay);
		}	
		
	}

	
	private void training(int maxIterations, Random rand, boolean visualize){
	    int SKIP_TICKS = 1000 / FRAMES_PER_SECOND;
	   
	    float next_game_tick = System.currentTimeMillis();
	    int curSeqID = 0;
	    
	    for (int i = 0; i < maxIterations; i++){
	    	rsom.flush();
	    	//Choose sequence
	    	curSeqID = rand.nextInt(sequences.size());
	    	SimpleMatrix[] curSequence = sequences.get(curSeqID);
	    	
	    	for (SimpleMatrix input : curSequence){
	    		//Temporal classification
	    		rsom.step((input));
	    		
	    		rsom.computeActivationMatrix();
	    		
	    		if (visualize){
		    		//Update graphics
		    		updateGraphics(input, curSeqID);
		    		
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
	    	
    		rsom.sensitize(i);
	    }
	    
	}

	private void setupVisualization(ISOM som, int GUI_SIZE){
		//Create GUI
		
		frame = new MovingLinesGUI_Prediction(som, rsom);
		frame.setTitle("Visualiztion");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		updateGraphics(blank,0); //Give a blank
		frame.pack();
		frame.setVisible(true);	
	}
	
	private void updateGraphics(SimpleMatrix inputVector, int iteration){
		spatialDummy.step(inputVector.getMatrix().data);
		frame.updateData(inputVector, spatialDummy, rsom);
		frame.setTitle("Visualiztion - Sequence: " + iteration);
		frame.revalidate();
		frame.repaint();

	}


	private void buildSequences(){
		createFigures();
		sequences = new ArrayList<SimpleMatrix[]>();

		SimpleMatrix[] seq1 = {bigT};
		SimpleMatrix[] seq2 = {bigO};
		SimpleMatrix[] seq3 = {smallO};
		
		SimpleMatrix[] seq4 = {smallO, bigO};
		SimpleMatrix[] seq5 = {bigT, smallO};
		SimpleMatrix[] seq6 = {bigT, bigT};
		
		SimpleMatrix[] seq7 = {bigO, smallO, smallO, smallV};
		SimpleMatrix[] seq8 = {bigT, bigO, smallV, bigT};
		SimpleMatrix[] seq9 = {smallO, bigT, bigT, bigO};
		SimpleMatrix[] seq10 = {bigT, smallO, bigO, smallO};
		SimpleMatrix[] seq11 = {bigT, bigT, bigT, bigT};
		SimpleMatrix[] seq12 = {smallO, smallO, bigO, bigO};
		
		SimpleMatrix[] seq5_1 = {bigT, smallO, bigT, bigT, bigT};
		SimpleMatrix[] seq5_2 = {bigT, smallO, smallO, smallO, smallO};
		SimpleMatrix[] seq5_3 = {smallV, smallV, bigO, smallV, smallV};
		SimpleMatrix[] seq5_4 = {bigT, bigO, bigO, bigO, smallV};

		/*
		sequences.add(seq1);
		sequences.add(seq2);
		sequences.add(seq3);
		*/
		
		//sequences.add(seq4);
		//sequences.add(seq5);
		//sequences.add(seq6);
		
		//sequences.add(seq7);
		//sequences.add(seq8);
		
		/*
		sequences.add(seq9);
		sequences.add(seq10);
		sequences.add(seq11);
		sequences.add(seq12);
		*/
		//sequences.add(seq12);
		
		
		sequences.add(seq5_1);
		sequences.add(seq5_2);
		sequences.add(seq5_3);
		sequences.add(seq5_4);
		
	}
	
	private void createFigures(){		
		double[][] bigTData = {
				{1,0,0,0,0}};
		bigT = new SimpleMatrix(bigTData);
		bigT.reshape(1, bigT.numCols() * bigT.numRows());
		
		double[][] smallOData = {
				{0,1,0,0,0}};
		smallO = new SimpleMatrix(smallOData);
		smallO.reshape(1, smallO.numCols() * smallO.numRows());
		
		double[][] bigOData = {
				{0,0,1,0,0}};
		bigO = new SimpleMatrix(bigOData);
		bigO.reshape(1, bigO.numCols() * bigO.numRows());
		
		double[][] smallVData = {
				{0,0,0,1,0}};
		smallV = new SimpleMatrix(smallVData);
		smallV.reshape(1, smallV.numCols() * smallV.numRows());
		
		double[][] blankData = {
				{0,0,0,0,1}};
		blank = new SimpleMatrix(blankData);
		blank.reshape(1, blank.numCols() * blank.numRows());
	}

}
