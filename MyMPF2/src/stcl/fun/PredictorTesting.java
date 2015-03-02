package stcl.fun;

import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.predictors.FirstOrderPredictor;

public class PredictorTesting {
	private SimpleMatrix a,b,c;
	private ArrayList<SimpleMatrix> possibleLetters;
	private ArrayList<SimpleMatrix[]> sequences;
	private FirstOrderPredictor predictor;
	private Random rand = new Random(1234);
	public PredictorTesting() {
		// TODO Auto-generated constructor stub
	}

	public static void main(String[] args) {
		PredictorTesting pred = new PredictorTesting();
		pred.run();

	}
	
	public void run(){
		setupExperiment();
		runTraining(100);
	}
	
	private void setupExperiment(){
		buildSequences();
		predictor = new FirstOrderPredictor(1,2);
	}
	
	private void runTraining(int iterations){
		for (int i = 0; i < iterations; i++){
			int seqID = rand.nextInt(sequences.size());
			
			SimpleMatrix[] sequence = sequences.get(seqID);
			
			for (SimpleMatrix m : sequence){
				SimpleMatrix output = predictor.predict(m, 1, true);
				System.out.println(calculateEntropy(output));				
			}
		}
	}
	
	private double calculateEntropy(SimpleMatrix m){
		double entropy = 0;
		int elements = m.getNumElements();
		
		for (double d : m.getMatrix().data){
			double value = d + 0.0000001; //Have to add small e to make sure that we don't get NaN when d = 0
			entropy += value * logOfBase(elements, value); 
		}
		
		return Math.abs(entropy);
	}
	
	private double logOfBase(int base, double num) {
	    return Math.log(num) / Math.log(base);
	}
	private void buildSequences(){
		buildLetters();
		sequences = new ArrayList<SimpleMatrix[]>();
		
		SimpleMatrix[] seq1 = {a,b,b};
		SimpleMatrix[] seq2 = {b,c,c};
		SimpleMatrix[] seq3 = {c,a,a};
		
		sequences.add(seq1);
		sequences.add(seq2);
		sequences.add(seq3);

		
		/*
		for (SimpleMatrix a : possibleLetters){
			for (SimpleMatrix b : possibleLetters){
				for (SimpleMatrix c : possibleLetters){
					SimpleMatrix[] seq = {a,b,c};
					sequences.add(seq);
				}
			}
		}
		*/
	}
	
	private void buildLetters(){
		possibleLetters = new ArrayList<SimpleMatrix>();
		double[][] a_data = {{0,0}};
		double[][] b_data = {{0,1}};
		double[][] c_data = {{1,0}};
		
		a = new SimpleMatrix(a_data);
		b = new SimpleMatrix(b_data);
		c = new SimpleMatrix(c_data);
		possibleLetters.add(a);
		possibleLetters.add(b);
		possibleLetters.add(c);
		
	}

}
