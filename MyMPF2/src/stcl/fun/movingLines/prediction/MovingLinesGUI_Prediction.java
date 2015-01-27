package stcl.fun.movingLines.prediction;

import java.awt.Dimension;
import java.awt.GridLayout;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.brain.NeoCorticalUnit;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.fun.movingLines.MatrixPanel;
import stcl.fun.movingLines.SomPanel;
import dk.stcl.som.containers.SomNode;
import dk.stcl.som.rsom.IRSOM;
import dk.stcl.som.som.ISOM;
import dk.stcl.som.som.SOM;

public class MovingLinesGUI_Prediction extends JFrame {
	private MatrixPanel input;
	private SomPanel spatialModels;
	private MatrixPanel spatialActivation;
	private MatrixPanel predictedInput;
	private SomPanel rsomModel1;
	private SomPanel rsomModel2;
	private SomPanel rsomModel3;
	private SomPanel rsomModel4;
	private MatrixPanel rsomActivation1;
	private MatrixPanel rsomActivation2;
	private MatrixPanel rsomActivation3;
	private MatrixPanel rsomActivation4;
	
	private int singleSomModelWidth;
	
	
	public MovingLinesGUI_Prediction(NeoCorticalUnit unit) {
		SOM spatialSom = unit.getSpatialPooler().getSOM();
		
		//Create overall grid layout
		int rows = 3;
		int cols = 4;
		int gap = 2;
		setLayout(new GridLayout(rows, cols, gap, gap));
		
		//Set preffered size of GUI
		int somHeight = spatialSom.getHeight(); //This should be equal to the width also
		singleSomModelWidth =  (int) Math.sqrt(spatialSom.getInputVectorLength()); //How many cells (size x size) are there in a single SOM model
		int somModelCellSize = 5;	  //How many pixels (size x size) does a single cell in a SOM model require in height and width
		int gapBetweenSomodels = 2;
		
		int somPanelSize = somHeight * singleSomModelWidth * gapBetweenSomodels * somModelCellSize;
		
		setPreferredSize(new Dimension(cols * somPanelSize, rows * somPanelSize));

		//Add input area
		input = new MatrixPanel(new SimpleMatrix(singleSomModelWidth, singleSomModelWidth));
		input.setSize(somHeight, somHeight);
		add(input);
		
		//Add visualization of spatial som models
		spatialModels = new SomPanel(spatialSom, somHeight, somHeight);
		spatialModels.setSize(somPanelSize, somPanelSize);
		add(spatialModels);
		
		//Add visualization of spatial activation
		spatialActivation = new MatrixPanel(new SimpleMatrix(5, 5));
		spatialActivation.setSize(somPanelSize, somPanelSize);
		add(spatialActivation);
		
		//Add predicted next input
		predictedInput = new MatrixPanel(new SimpleMatrix(singleSomModelWidth, singleSomModelWidth));
		predictedInput.setSize(somHeight, somHeight);
		add(predictedInput);
		
		// 2 x RSOM models
		rsomModel1 = new SomPanel(spatialSom, somHeight, somHeight);
		rsomModel1.setSize(somPanelSize, somPanelSize);
		add(rsomModel1);
		
		rsomModel2 = new SomPanel(spatialSom, somHeight, somHeight);
		rsomModel2.setSize(somPanelSize, somPanelSize);
		add(rsomModel2);
		
		//Add two of the RSOM activation cells
		rsomActivation1 = new MatrixPanel(new SimpleMatrix(1, 1));
		rsomActivation1.setSize(somPanelSize, somPanelSize);
		add(rsomActivation1);
		
		rsomActivation2 = new MatrixPanel(new SimpleMatrix(1, 1));
		rsomActivation2.setSize(somPanelSize, somPanelSize);
		add(rsomActivation2);
		
		// 2 x RSOM models
		rsomModel3 = new SomPanel(spatialSom, somHeight, somHeight);
		rsomModel3.setSize(somPanelSize, somPanelSize);
		add(rsomModel3);
		
		rsomModel4 = new SomPanel(spatialSom, somHeight, somHeight);
		rsomModel4.setSize(somPanelSize, somPanelSize);
		add(rsomModel4);
		
		//Add two of the RSOM activation cells
		rsomActivation3 = new MatrixPanel(new SimpleMatrix(1, 1));
		rsomActivation3.setSize(somPanelSize, somPanelSize);
		add(rsomActivation3);
		
		rsomActivation4 = new MatrixPanel(new SimpleMatrix(1, 1));
		rsomActivation4.setSize(somPanelSize, somPanelSize);
		add(rsomActivation4);		
	}
	
	public void updateData(SimpleMatrix inputVector, NeoCorticalUnit unit){
		
		SpatialPooler spatialPooler = unit.getSpatialPooler();
		TemporalPooler temporalPooler = unit.getTemporalPooler();
		
		//Update input area
		SimpleMatrix inputMatrix = new SimpleMatrix(inputVector);
		inputMatrix.reshape(singleSomModelWidth, singleSomModelWidth);
		input.updateData(inputMatrix);
		input.repaint();
		input.revalidate();
		
		//Update spatial activation
		SimpleMatrix activationMatrix = spatialPooler.getActivationMatrix();
		spatialActivation.updateData(activationMatrix);
		spatialActivation.repaint();
		spatialActivation.revalidate();
		
		//Get predicted next output
		SimpleMatrix expectedInput = unit.getFbOutput();
		expectedInput.reshape(singleSomModelWidth, singleSomModelWidth);
		predictedInput.updateData(expectedInput);
		predictedInput.repaint();
		predictedInput.revalidate();		
		
		//Create list of spatial models to be highlighted
		int maxID = findIDOfMaxValue(activationMatrix);
		boolean[] highlights = new boolean[activationMatrix.getMatrix().data.length];
		highlights[maxID] = true;
		
		//Update Spatial models
		SOM spatialSom = spatialPooler.getSOM();
		spatialModels.updateData(spatialSom, highlights);
		spatialModels.repaint();
		spatialModels.revalidate();		
		
		//Update RSOM activation
		SimpleMatrix temporalActivationMatrix = temporalPooler.getActivationMatrix();
		double tmp1[][] = {{temporalActivationMatrix.get(0, 0)}};
		rsomActivation1.updateData(new SimpleMatrix(tmp1));
		rsomActivation1.repaint();
		rsomActivation1.revalidate();
		
		double tmp2[][] = {{temporalActivationMatrix.get(0, 1)}};
		rsomActivation2.updateData(new SimpleMatrix(tmp2));
		rsomActivation2.repaint();
		rsomActivation2.revalidate();
		
		double tmp3[][] = {{temporalActivationMatrix.get(1, 0)}};
		rsomActivation3.updateData(new SimpleMatrix(tmp3));
		rsomActivation3.repaint();
		rsomActivation3.revalidate();
		
		double tmp4[][] = {{temporalActivationMatrix.get(1, 1)}};
		rsomActivation4.updateData(new SimpleMatrix(tmp4));
		rsomActivation4.repaint();
		rsomActivation4.revalidate();
		
		//Update RSOM models
		rsomModel1.updateData(spatialSom, getSpatialModelsInTemporalModel(temporalPooler, 0));
		rsomModel1.repaint();
		rsomModel1.revalidate();
		
		rsomModel2.updateData(spatialSom, getSpatialModelsInTemporalModel(temporalPooler, 1));
		rsomModel2.repaint();
		rsomModel2.revalidate();
		
		rsomModel3.updateData(spatialSom, getSpatialModelsInTemporalModel(temporalPooler, 2));
		rsomModel3.repaint();
		rsomModel3.revalidate();
		
		rsomModel4.updateData(spatialSom, getSpatialModelsInTemporalModel(temporalPooler, 3));
		rsomModel4.repaint();
		rsomModel4.revalidate();
		
		
		
	}
	
	private boolean[] getSpatialModelsInTemporalModel(TemporalPooler temporalPooler, int modelID){
		//Collect Model
		SomNode model = temporalPooler.getSOM().getNode(modelID);
		
		//Create boolean vector where an item is true if weight value of that item is higher than the mean.
		SimpleMatrix weightVector = model.getVector();
		int vectorSize = weightVector.numCols() * weightVector.numRows();
		double mean = weightVector.elementSum() / (double)vectorSize;
		double threshold = 0.9; //2 * ( 1 / (temporalPooler.getHeight() * temporalPooler.getWidth())); 
		boolean[] importantModels = new boolean[vectorSize];
		
		for (int i = 0; i <vectorSize; i++){
			if (weightVector.get(i) > threshold){
				importantModels[i] = true;
			}
		}
		
		
		return importantModels;
	}
	
	/**
	 * 
	 * @param m
	 * @return id of cell with max value
	 */
	private int findIDOfMaxValue(SimpleMatrix m){
		double maxValue = Double.NEGATIVE_INFINITY;
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
		return maxID;
	}

	

}
