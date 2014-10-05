package stcl.fun.movingLines;

import java.awt.Dimension;
import java.awt.GraphicsConfiguration;
import java.awt.GridLayout;
import java.awt.HeadlessException;

import javax.swing.JFrame;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.algo.som.SOM;

public class MovingLinesGUI extends JFrame {
	private MatrixPanel input;
	private SomPanel spatialModels;
	private MatrixPanel spatialActivation;
	private SomPanel possibleInputs;
	private SomPanel rsomModel1;
	private SomPanel rsomModel2;
	private SomPanel rsomModel3;
	private SomPanel rsomModel4;
	private MatrixPanel rsomActivation1;
	private MatrixPanel rsomActivation2;
	private MatrixPanel rsomActivation3;
	private MatrixPanel rsomActivation4;
	
	public MovingLinesGUI(SOM spatialSom, SOM possibleInputsSom) {
		//Create overall grid layout
		int rows = 3;
		int cols = 4;
		int gap = 2;
		setLayout(new GridLayout(rows, cols, gap, gap));
		
		//Set preffered size of GUI
		int somModelSize = 3;
		int somModelCellSize = 10;
		int numberOfModelsInSOMS = 5;
		int gapBetweenSomodels = 2;
		int somSize = somModelSize * numberOfModelsInSOMS * gapBetweenSomodels * somModelCellSize;
		
		setPreferredSize(new Dimension(cols * somSize, rows * somSize));
		
		
		//Add input area
		input = new MatrixPanel(new SimpleMatrix(3, 3));
		input.setSize(somModelSize, somModelSize);
		add(input);
		
		//Add visualization of spatial som models
		spatialModels = new SomPanel(spatialSom, somModelSize, somModelSize);
		spatialModels.setSize(somSize, somSize);
		add(spatialModels);
		
		//Add visualization of spatial activation
		spatialActivation = new MatrixPanel(new SimpleMatrix(5, 5));
		spatialActivation.setSize(somSize, somSize);
		add(spatialActivation);
		
		//Add blank (Or the different figures that can be seen
		possibleInputs = new SomPanel(possibleInputsSom, somModelSize, somModelSize);
		possibleInputs.setSize(somSize, somSize);
		add(possibleInputs);
		
		// 2 x RSOM models
		rsomModel1 = new SomPanel(spatialSom, somModelSize, somModelSize);
		rsomModel1.setSize(somSize, somSize);
		add(rsomModel1);
		
		rsomModel2 = new SomPanel(spatialSom, somModelSize, somModelSize);
		rsomModel2.setSize(somSize, somSize);
		add(rsomModel2);
		
		//Add two of the RSOM activation cells
		rsomActivation1 = new MatrixPanel(new SimpleMatrix(1, 1));
		rsomActivation1.setSize(somSize, somSize);
		add(rsomActivation1);
		
		rsomActivation2 = new MatrixPanel(new SimpleMatrix(1, 1));
		rsomActivation2.setSize(somSize, somSize);
		add(rsomActivation2);
		
		// 2 x RSOM models
		rsomModel3 = new SomPanel(spatialSom, somModelSize, somModelSize);
		rsomModel3.setSize(somSize, somSize);
		add(rsomModel3);
		
		rsomModel4 = new SomPanel(spatialSom, somModelSize, somModelSize);
		rsomModel4.setSize(somSize, somSize);
		add(rsomModel4);
		
		//Add two of the RSOM activation cells
		rsomActivation3 = new MatrixPanel(new SimpleMatrix(1, 1));
		rsomActivation3.setSize(somSize, somSize);
		add(rsomActivation3);
		
		rsomActivation4 = new MatrixPanel(new SimpleMatrix(1, 1));
		rsomActivation4.setSize(somSize, somSize);
		add(rsomActivation4);		
	}
	
	public void updateData(SimpleMatrix inputVector, SpatialPooler spatialPooler, TemporalPooler temporalPooler){
		
		//Update input area
		SimpleMatrix inputMatrix = new SimpleMatrix(inputVector);
		inputMatrix.reshape(3, 3);
		input.updateData(inputMatrix);
		input.repaint();
		input.revalidate();
		
		//Update spatial activation
		SimpleMatrix activationMatrix = spatialPooler.getActivationMatrix();
		spatialActivation.updateData(activationMatrix);
		spatialActivation.repaint();
		spatialActivation.revalidate();
		
		
		//Create list of spatial models to be highlighted
		int maxID = findIDOfMaxValue(activationMatrix);
		boolean[] highlights = new boolean[activationMatrix.getMatrix().data.length];
		highlights[maxID] = true;
		
		//Update Spatial models
		SOM spatialSom = spatialPooler.getSOM();
		spatialModels.updateData(spatialSom, highlights);
		//spatialModels.updateData(spatialSom);
		spatialModels.repaint();
		spatialModels.revalidate();		
		
		
		//Update RSOM models
		rsomModel1.updateData(spatialSom);
		rsomModel1.repaint();
		rsomModel1.revalidate();
		
		rsomModel2.updateData(spatialSom);
		rsomModel2.repaint();
		rsomModel2.revalidate();
		
		rsomModel3.updateData(spatialSom);
		rsomModel3.repaint();
		rsomModel3.revalidate();
		
		rsomModel4.updateData(spatialSom);
		rsomModel4.repaint();
		rsomModel4.revalidate();
		
		
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
