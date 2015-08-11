package stcl.graphics;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.GridLayout;

import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingConstants;

import org.ejml.simple.SimpleMatrix;

import dk.stcl.gui.SimpleMatrixVisualizer;

public class NetworkPanel extends JPanel {
	private static Color BACKGROUND = Color.red;
	
	private SimpleMatrixVisualizer inputObservation, inputAction;
	private SimpleMatrixVisualizer outputPrediction, outputAction;
	
	public void initialize(int observationMatrixSize, int actionMatrixSize){
		
		Box mainPanel = new Box(BoxLayout.X_AXIS);
		mainPanel.add(Box.createGlue());
		
		JPanel inputPanel = new JPanel();
		inputPanel.setLayout(new GridLayout(1, 2, 10, 0));
		
		Box inputObservationArea = new Box(BoxLayout.Y_AXIS);
		JLabel lblInputObservation = new JLabel("Input");
		inputObservation = new SimpleMatrixVisualizer();
		inputObservation.initialize(new SimpleMatrix(observationMatrixSize, observationMatrixSize), true);
		setLayoutOfSubArea(inputObservationArea, lblInputObservation, inputObservation);
		
		inputPanel.add(inputObservationArea);
		
		
		Box inputActionArea = new Box(BoxLayout.Y_AXIS);
		JLabel lblInputAction = new JLabel("Action");
		inputAction = new SimpleMatrixVisualizer();
		inputAction.initialize(new SimpleMatrix(actionMatrixSize, actionMatrixSize), true);
		setLayoutOfSubArea(inputActionArea, lblInputAction, inputAction);
		inputPanel.add(inputActionArea);
		
		mainPanel.add(inputPanel);
		
		JPanel outputPanel = new JPanel();
		inputPanel.setLayout(new GridLayout(1, 2, 10, 0));
		
		Box outputObservationArea = new Box(BoxLayout.Y_AXIS);
		JLabel lblOutputObservation = new JLabel("Prediction");
		outputPrediction = new SimpleMatrixVisualizer();
		outputPrediction.initialize(new SimpleMatrix(observationMatrixSize, observationMatrixSize), true);
		setLayoutOfSubArea(outputObservationArea, lblOutputObservation, outputPrediction);
		outputPanel.add(outputObservationArea);
		
		Box outputActionArea = new Box(BoxLayout.Y_AXIS);
		JLabel lblOutputAction= new JLabel("Next action");
		outputAction = new SimpleMatrixVisualizer();
		outputAction.initialize(new SimpleMatrix(actionMatrixSize, actionMatrixSize), true);
		setLayoutOfSubArea(outputActionArea, lblOutputAction, outputAction);		
		outputPanel.add(outputActionArea);
		
		mainPanel.add(outputPanel);
		
		this.add(mainPanel);
	}
	
	public void updateData(SimpleMatrix input, SimpleMatrix actionNow, SimpleMatrix prediction, SimpleMatrix actionNext){
		inputObservation.registerMatrix(input);
		inputAction.registerMatrix(actionNow);
		outputPrediction.registerMatrix(prediction);
		outputAction.registerMatrix(actionNext);
	}
	
	private void setLayoutOfSubArea(Box area, JLabel label, JPanel panel){
		//area.setLayout(new BorderLayout());
		panel.setBackground(BACKGROUND);
		area.setBackground(BACKGROUND);
		panel.setPreferredSize(new Dimension(250, 250));
		label.setHorizontalAlignment(SwingConstants.CENTER);
		area.add(label, BorderLayout.NORTH);
		area.add(panel, BorderLayout.CENTER);
	}
}
