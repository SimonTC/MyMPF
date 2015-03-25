package stcl.algo.brain;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.NewSequencer;
import stcl.algo.poolers.SOM;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.algo.predictors.ActionDecider;
import stcl.algo.predictors.Decider;
import stcl.algo.predictors.Predictor;
import stcl.algo.predictors.Predictor_VOMM;
import stcl.algo.util.Normalizer;
import stcl.algo.util.Orthogonalizer;
import dk.stcl.core.basic.containers.SomNode;

public class NeoCorticalUnit{
	
	private SpatialPooler spatialPooler;
	private Predictor_VOMM predictor;
	//private Decider decider;
	private ActionDecider actionDecider;
	private SimpleMatrix biasMatrix;
	private SimpleMatrix nextActionVotes;
	private SimpleMatrix ffInput;
	private SimpleMatrix fbInput;
	
	private SimpleMatrix ffOutput;
	private SimpleMatrix fbOutput;
	
	private int ffInputVectorSize;
	private int spatialMapSize;
	private int temporalMapSize;
	
	private boolean needHelp;
	private boolean neededHelpThisTurn; //Used in reporting
	private double predictionEntropy;
	private double entropyThreshold; //The exponential moving average of the prediction entropy
	private double entropyDiscountingFactor;
	
	private boolean usePrediction;
	private boolean active;
	
	private Random rand;

	private boolean entropyThresholdFrozen;
	private boolean biasBeforePredicting;
	private boolean useBiasedInputInSequencer;
	
	private NewSequencer sequencer;
	private boolean noTemporal;
	
	private int wantedNextAction;
	
	
	public NeoCorticalUnit(Random rand, int ffInputLength, int spatialMapSize, int temporalMapSize, double initialPredictionLearningRate, boolean useMarkovPrediction, int markovOrder) {
		this(rand, ffInputLength, spatialMapSize, temporalMapSize, initialPredictionLearningRate, useMarkovPrediction, markovOrder, false, 1); //TODO: Not good with actionMatrix size. Has to be changed in future
	}

	/**
	 * 
	 * @param rand
	 * @param ffInputLength
	 * @param spatialMapSize
	 * @param temporalMapSize
	 * @param initialPredictionLearningRate
	 * @param useMarkovPrediction
	 * @param markovOrder
	 * @param noTemporal
	 */
	public NeoCorticalUnit(Random rand, int ffInputLength, int spatialMapSize, int temporalMapSize, double initialPredictionLearningRate, boolean useMarkovPrediction, int markovOrder, boolean noTemporal, int actionMatrixSize) {
		double decay = calculateDecay(markovOrder,0.01);// 1.0 / markovOrder);
		entropyDiscountingFactor = decay; //TODO: Does this make sense?
		//TODO: All parameters should be handled in parameter file
		
		this.rand = rand;
		
		spatialPooler = new SpatialPooler(rand, ffInputLength, spatialMapSize, 0.1, Math.sqrt(spatialMapSize), 0.125); //TODO: Move all parameters out
		if (!noTemporal) {
			sequencer = new NewSequencer(markovOrder, temporalMapSize, spatialMapSize * spatialMapSize, actionMatrixSize, decay);
			this.temporalMapSize = temporalMapSize;
		} else {
			this.temporalMapSize = spatialMapSize;
		}
		actionDecider = new ActionDecider(markovOrder, 0.1, rand, actionMatrixSize);
		//decider = new Decider(markovOrder, initialPredictionLearningRate, rand, 1, 1, 0.3, spatialMapSize);
		predictor = new Predictor_VOMM(markovOrder, initialPredictionLearningRate, rand);
		biasMatrix = new SimpleMatrix(spatialMapSize, spatialMapSize);
		biasMatrix.set(1);
		ffOutput = new SimpleMatrix(this.temporalMapSize, this.temporalMapSize);
		fbOutput = new SimpleMatrix(1, ffInputLength);
		ffInputVectorSize = ffInputLength;
		this.usePrediction = useMarkovPrediction;
		this.spatialMapSize = spatialMapSize;
		nextActionVotes = new SimpleMatrix(spatialMapSize, spatialMapSize);
		nextActionVotes.set(1);
		nextActionVotes = Normalizer.normalize(nextActionVotes);
		
		needHelp = false;
		entropyThreshold = 0;
		entropyThresholdFrozen = false;
		biasBeforePredicting = false;
		useBiasedInputInSequencer = false;
		this.noTemporal = noTemporal;
		wantedNextAction = -1;
	}
	
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		return this.feedForward(inputVector, 0,-1);
	}
	
	public SimpleMatrix feedForward(SimpleMatrix inputVector, double reward, int actionPerformed){
		//Test input
		if (!inputVector.isVector()) throw new IllegalArgumentException("The feed forward input to the neocortical unit has to be a vector");
		if (inputVector.numCols() != ffInputVectorSize) throw new IllegalArgumentException("The feed forward input to the neocortical unit has to be a 1 x " + ffInputVectorSize + " vector");
		
		active = true;
		
		ffInput = inputVector;
		
		//Spatial classification
		SimpleMatrix spatialFFOutputMatrix = spatialPooler.feedForward(inputVector);
		
		//Bias output
		//SimpleMatrix biasedOutput = biasMatrix(spatialFFOutputMatrix, biasMatrix);
		//SimpleMatrix biasedOutput = biasTowardsPrediction(spatialFFOutputMatrix, biasMatrix, 0.5);
		SimpleMatrix biasedOutput  = spatialFFOutputMatrix;
		
		ffOutput = biasedOutput;
		needHelp = true;
		SimpleMatrix prediction;
		if (!noTemporal) {
			double[] spatialFFOutputDataVector;
			if (useBiasedInputInSequencer){
				spatialFFOutputDataVector = biasedOutput.getMatrix().data;		
			} else {
				spatialFFOutputDataVector = spatialFFOutputMatrix.getMatrix().data;	
			}
			SimpleMatrix temporalFFInputVector = new SimpleMatrix(1, spatialFFOutputDataVector.length);
			temporalFFInputVector.getMatrix().data = spatialFFOutputDataVector;
			
			ffOutput = sequencer.feedForward(temporalFFInputVector, reward, actionPerformed);
			
			needHelp = sequencer.needHelp();
			
			wantedNextAction = sequencer.getWantedAction();
		} else {
			//ffOutput = Orthogonalizer.aggressiveOrthogonalization(ffOutput);
		}
		neededHelpThisTurn = needHelp;
		
		return ffOutput;
	}
	
	
	/**
	 * 
	 * @param inputMatrix
	 * @param correlationMatrix
	 * @return
	 */
	public SimpleMatrix feedBackward(SimpleMatrix inputMatrix, int chosenAction){
		//Test input
		//if (inputMatrix.isVector()) throw new IllegalArgumentException("The feed back input to the neocortical unit has to be a matrix");
		if (inputMatrix.numCols() != temporalMapSize || inputMatrix.numRows() != temporalMapSize) throw new IllegalArgumentException("The feed back input to the neocortical unit has to be a " + temporalMapSize + " x " + temporalMapSize + " matrix");

		fbInput = inputMatrix;
		
		biasMatrix = sequencer.feedBackward(inputMatrix, chosenAction);
		
		//biasMatrix = biasMatrix.plus(0.1 / biasMatrix.getNumElements()); //Add small uniform mass
		
		SimpleMatrix biasedTemporalFBOutput = new SimpleMatrix(biasMatrix);
		biasedTemporalFBOutput.reshape(spatialMapSize, spatialMapSize);
		
		//Selection of best spatial mode
		SimpleMatrix spatialPoolerFBOutputVector = spatialPooler.feedBackward(biasedTemporalFBOutput);
		
		fbOutput = spatialPoolerFBOutputVector;
		
		//fbOutput = addNoise(fbOutput, 0.1);
		//fbOutput = Normalizer.normalize(fbOutput);
		
		needHelp = false;
		active = false;
		
		return fbOutput;
	}
	
	/**
	 * Adds noise to the given matrix and returns the matrix.
	 * The matrix is altered in this method.
	 * @param m
	 * @param noiseMagnitude
	 * @return
	 */
	private SimpleMatrix addNoise(SimpleMatrix m, double noiseMagnitude){
		double noise = (rand.nextDouble() - 0.5) * 2 * noiseMagnitude;
		m = m.plus(noise);
		for (int i = 0; i < m.getNumElements(); i++){
			double d = m.get(i);
			d = d + (rand.nextDouble() - 0.5) * 2 * noiseMagnitude;
			if (d < 0) d = 0;
			m.set(i, d);
		}
		return m;
	}
	
	/**
	 * Calculate the decay which will have the first input in a sequence have a t least minInfluence influence on the difference vector in the rsom
	 * @param memoryLength
	 * @param minInfluence
	 * @return
	 */
	private double calculateDecay(int memoryLength, double minInfluence){
		double decay = 1 - Math.pow(minInfluence, 1.0 / memoryLength);
		return decay;
 	}
	
	/**
	 * Bias the matrixToBias by element multiplying it with the biasMatrix
	 * @param matrixToBias
	 * @param biasMatrix
	 * @return
	 */
	private SimpleMatrix biasMatrix(SimpleMatrix matrixToBias, SimpleMatrix biasMatrix){
		SimpleMatrix biasedMatrix = matrixToBias.elementMult(biasMatrix);
		biasedMatrix = Normalizer.normalize(biasedMatrix);
		return biasedMatrix;
	}
	
	/**
	 * Bias the matrixToBias by element adding it with the biasMatrix. The values in the biasmatrix are only influencing by predictionInfluence
	 * @param matrixToBias
	 * @param predictionMatrix
	 * @param predictionInfluence
	 * @return
	 */
	private SimpleMatrix biasTowardsPrediction(SimpleMatrix matrixToBias, SimpleMatrix predictionMatrix, double predictionInfluence){
		SimpleMatrix biasedMatrix = matrixToBias.plus(predictionInfluence, predictionMatrix);
		biasedMatrix = Normalizer.normalize(biasedMatrix);
		return biasedMatrix;
	}
	
	private double calculateEntropy(SimpleMatrix m){
		double sum = 0;
		for (Double d : m.getMatrix().data){
			if (d != 0) sum += d * Math.log(d);
		}
		return -sum;
	}
	
	private SimpleMatrix normalize(SimpleMatrix m){
		return Normalizer.normalize(m);
	}
	
	public void flush(){
		biasMatrix.set(1);
		actionDecider.flush();
		predictor.flush();
		if (sequencer != null) sequencer.reset();
	}
	
	public void setLearning(boolean learning){
		spatialPooler.setLearning(learning);
		actionDecider.setLearning(learning);
		if (sequencer != null) sequencer.setLearning(learning);
	}

	public SpatialPooler getSpatialPooler() {
		return spatialPooler;
	}

	public SimpleMatrix getFFOutput() {
		return ffOutput;
	}

	public SimpleMatrix getFBOutput() {
		return fbOutput;
	}
	
	public void sensitize(int iteration){
		spatialPooler.sensitize(iteration);
	}
	
	public int findTemporalBMUID(){
		int maxID = -1;
		double maxValue = Double.NEGATIVE_INFINITY;
		
		for (int i = 0; i < ffOutput.getNumElements(); i++){
			double v = ffOutput.get(i);
			if (v > maxValue){
				maxValue = v;
				maxID = i;
			}
		}
		
		return maxID;
	}
	
	/*
	public Predictor getPredictor(){
		return predictor;
	}
*/
	public SOM getSOM() {
		return spatialPooler.getSOM();
	}

	public void printPredictionModel() {
		actionDecider.printModel();
		
	}

	public boolean needHelp() {
		return needHelp;
	}
	
	public void setNeedHelp(boolean needHelp){
		//this.needHelp = needHelp;
		//neededHelpThisTurn = needHelp;
	}

	public double getEntropy() {
		return predictionEntropy;
	}
	
	public double getEntropyThreshold() {
		return entropyThreshold;
	}
	
	public boolean active(){
		return active;
	}
	
	public NewSequencer getSequencer(){
		return sequencer;
	}

	/**
	 * @param entropyThresholdFrozen the entropyThresholdFrozen to set
	 */
	public void setEntropyThresholdFrozen(boolean entropyThresholdFrozen) {
		this.entropyThresholdFrozen = entropyThresholdFrozen;
	}
	
	public void setBiasBeforePrediction(boolean flag){
		biasBeforePredicting = flag;
	}

	public void setUseBiasedInputInSequencer(boolean useBiasedInputInSequencer) {
		this.useBiasedInputInSequencer = useBiasedInputInSequencer;
	}

	/**
	 * @return the temporalMapSize
	 */
	public int getTemporalMapSize() {
		return temporalMapSize;
	}
	
	public void setUsePrediction(boolean usePrediction){
		this.usePrediction = usePrediction;
	}
	
	public SimpleMatrix getFFInput(){
		return ffInput;
	}
	
	public SimpleMatrix getFBInput(){
		return fbInput;
	}
	
	public ActionDecider getDecider(){
		return actionDecider;
	}
	
	public int getActionVote(){
		return wantedNextAction;
	}
	

}
