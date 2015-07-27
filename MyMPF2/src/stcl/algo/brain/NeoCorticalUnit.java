package stcl.algo.brain;

import java.io.Serializable;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import stcl.algo.poolers.Sequencer;
import stcl.algo.poolers.SOM;
import stcl.algo.poolers.SpatialPooler;
import stcl.algo.poolers.TemporalPooler;
import stcl.algo.predictors.Predictor_VOMM;
import stcl.algo.util.Normalizer;
import stcl.algo.util.Orthogonalizer;
import dk.stcl.core.basic.containers.SomNode;
import dk.stcl.core.utils.SomConstants;

public class NeoCorticalUnit implements Serializable{
	private static final long serialVersionUID = 1L;
	private SpatialPooler spatialPooler;
	private Predictor_VOMM predictor;
	private ActionDecider_Q decider;
	private SimpleMatrix biasMatrix;
	private SimpleMatrix predictionMatrix;
	private SimpleMatrix ffInput;
	private SimpleMatrix fbInput;
	
	private SimpleMatrix ffOutput;
	private SimpleMatrix fbOutput;
	
	private int ffInputVectorSize;
	private int spatialMapSize;
	private int temporalMapSize;
	private int ffOutputMapSize;
	
	private boolean needHelp;
	private boolean neededHelpThisTurn; //Used in reporting
	private double predictionEntropy;
	private double entropyThreshold; //The exponential moving average of the prediction entropy
	private double entropyDiscountingFactor;
	
	private boolean usePrediction;
	private boolean active;
	
	private Random rand;

	private boolean entropyThresholdFrozen;
	
	private TemporalPooler temporalPooler;
	//private Sequencer sequencer;
	private boolean noTemporal;
	private boolean noSpatial;
	
	private int chosenAction; 
	private int markovOrder;
	
	private int numPossibleActions;
	private boolean reactionary;
	private boolean offlineLearning;
	
	private int stepsSinceSequenceStart;
	private SimpleMatrix temporalProbabilityMatrixToSend;

	/**
	 * If true the FF output from the spatial pooler will be biased by the prediction done at t-1
	 */
	private boolean biasSpatialFFOutput; 
	
	private boolean useBiasedInputInPredictor;
	private boolean useBiasedInputInSequencer;
	private boolean useBiasedInputInDecider;

	/**
	 * 
	 * @param rand
	 * @param ffInputLength
	 * @param spatialMapSize
	 * @param temporalMapSize
	 * @param markovOrder
	 * @param numPossibleActions
	 * @param usePrediction
	 * @param reactionary
	 * @param offlineLearning
	 */
	public NeoCorticalUnit(Random rand, int ffInputLength, int spatialMapSize, int temporalMapSize, int markovOrder, int numPossibleActions, boolean usePrediction, boolean reactionary, boolean offlineLearning) {
		initialize(rand, ffInputLength, spatialMapSize, temporalMapSize, markovOrder, numPossibleActions, usePrediction, reactionary, offlineLearning);
	}
	
	public NeoCorticalUnit(String initializationString, Random rand){
		String[] lines = initializationString.split(SomConstants.LINE_SEPARATOR);
		String[] unitInfo = lines[0].split(" ");
		initialize(rand, Integer.parseInt(unitInfo[0]), Integer.parseInt(unitInfo[1]), Integer.parseInt(unitInfo[2]), Integer.parseInt(unitInfo[3]), Integer.parseInt(unitInfo[4]), Boolean.parseBoolean(unitInfo[5]), Boolean.parseBoolean(unitInfo[6]), Boolean.parseBoolean(unitInfo[7]));
		if (!noSpatial) spatialPooler = new SpatialPooler(initializationString, 1, rand);
		if (!noTemporal){
			int temporalStart = 0;
			String[] tmp = initializationString.split(SomConstants.LINE_SEPARATOR);
			for (int i = 0; i < tmp.length; i++){
				if (tmp[i].equalsIgnoreCase("TEMPORAL")){
					temporalStart = i + 1;
					break;
				}
			}
			temporalPooler = new TemporalPooler(initializationString, temporalStart, rand);
		}
	}
	public String toInitializationString(){
		String s = ffInputVectorSize + " " + spatialMapSize + " " + temporalMapSize + " " + markovOrder + " " + numPossibleActions + " " + usePrediction + " " + reactionary + " " + offlineLearning + SomConstants.LINE_SEPARATOR;
		if (!noSpatial) s+= spatialPooler.toInitializationString();
		if (!noTemporal) s+= "TEMPORAL" + SomConstants.LINE_SEPARATOR +  temporalPooler.toInitializationString();
		return s;
	}
	
	private void initialize(Random rand, int ffInputLength, int spatialMapSize, int temporalMapSize, int markovOrder, int numPossibleActions, boolean usePrediction, boolean reactionary, boolean offlineLearning){
		//Test arguments
		if (ffInputLength < 1) throw new IllegalArgumentException("Input length has to be greater than 0");
		if (usePrediction && markovOrder < 1) throw new IllegalArgumentException("Markov order has to be greater than 0 when using prediction");
		if (markovOrder < 1 && temporalMapSize > 0) throw new IllegalArgumentException("Markov order has to be greater than 0 when using the temporal pooler");
		if (spatialMapSize < 1 && temporalMapSize < 1) throw new IllegalArgumentException("The spatial pooler and temporal pooler can't both have a map size of 0");
		
		//Instantiate sub-components
		int spatialOutputLength = ffInputLength;
		noSpatial = spatialMapSize < 1;
		if (!noSpatial) {
			spatialPooler = instantiateSpatialPooler(rand, ffInputLength, spatialMapSize, 0.1, Math.sqrt(spatialMapSize), 0.125); //TODO: Move all parameters out
			spatialOutputLength = (int) Math.pow(spatialMapSize, 2);
		}
		
		if (numPossibleActions > 0) decider = instantiateActionDecider(numPossibleActions, spatialOutputLength, 0.9, offlineLearning, reactionary); //TODO: Move all parameters out
		
		if (markovOrder > 0) predictor = instantiatePredictor(markovOrder, 0.1, rand); //TODO: Move all parameters out
		
		double decay = calculateDecay(markovOrder,0.01);// 1.0 / markovOrder);

		noTemporal = (temporalMapSize < 1);
		if (temporalMapSize > 0) temporalPooler = instantiateTemporalPooler(rand, spatialOutputLength, temporalMapSize, 0.1, Math.sqrt(temporalMapSize), 0.125, decay);
		
		//Set map and output sizes
		this.spatialMapSize = spatialMapSize;
		this.temporalMapSize = temporalMapSize;
		
		ffOutputMapSize = noTemporal ? spatialMapSize : temporalMapSize;
		
		ffOutput = new SimpleMatrix(this.ffOutputMapSize, this.ffOutputMapSize);
		fbOutput = new SimpleMatrix(1, ffInputLength);
		ffInputVectorSize = ffInputLength;
		
		
		
		//Set flags
		this.usePrediction = usePrediction;
		needHelp = false;
		entropyThresholdFrozen = false;
		biasSpatialFFOutput = false; //TODO: Make sure this can be changed
		useBiasedInputInPredictor = false;
		useBiasedInputInSequencer = false;
		
		//Set fields
		
		entropyDiscountingFactor = decay; //TODO: Does this make sense?
		this.rand = rand;
		entropyThreshold = 0;
		this.markovOrder = markovOrder;
		chosenAction = -1;
		this.numPossibleActions = numPossibleActions; 
		this.reactionary = reactionary;
		this.offlineLearning = offlineLearning;
		this.temporalMapSize = temporalMapSize;
		stepsSinceSequenceStart = 0;
		temporalProbabilityMatrixToSend = new SimpleMatrix(ffOutputMapSize, ffOutputMapSize);
		
		//Initialize matrices
		if (noSpatial){
			biasMatrix = new SimpleMatrix(1, ffInputLength);
			predictionMatrix = new SimpleMatrix(1, ffInputLength);
		} else {
			biasMatrix = new SimpleMatrix(spatialMapSize, spatialMapSize);
			predictionMatrix = new SimpleMatrix(spatialMapSize, spatialMapSize);
		}
		biasMatrix.set(1);
		predictionMatrix.set(1);
		predictionMatrix = Normalizer.normalize(predictionMatrix);
		
	}
	
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		return this.feedForward(inputVector, 0,0);
	}
	
	public SimpleMatrix feedForward(SimpleMatrix inputVector, double reward, int actionPerformed){
		//Test input
		if (!inputVector.isVector()) throw new IllegalArgumentException("The feed forward input to the neocortical unit has to be a vector");
		if (inputVector.numCols() != ffInputVectorSize) throw new IllegalArgumentException("The feed forward input to the neocortical unit has to be a 1 x " + ffInputVectorSize + " vector");
		
		active = true;
		
		ffInput = inputVector;
		
		//Spatial classification
		SimpleMatrix spatialFFOutputMatrix = ffInput;
		if (!noSpatial) spatialFFOutputMatrix = spatialPooler.feedForward(inputVector);
		SimpleMatrix biasedSpatialFFOutputMatrix = biasMatrix(spatialFFOutputMatrix, biasMatrix);
		
		needHelp = true;
		
		if (decider != null){
			SimpleMatrix inputToDecider = useBiasedInputInDecider ? biasedSpatialFFOutputMatrix : spatialFFOutputMatrix;
			feedDecider(inputToDecider, actionPerformed, reward);
		}
		
		if (predictor != null){
			if (usePrediction){
				SimpleMatrix inputToPredictor = useBiasedInputInPredictor ? biasedSpatialFFOutputMatrix : spatialFFOutputMatrix;
				predictionMatrix = predictor.predict(inputToPredictor);			
				predictionEntropy = calculateEntropy(predictionMatrix);			
				needHelp =  (predictionEntropy > entropyThreshold);
				if (!entropyThresholdFrozen){
					entropyThreshold = entropyDiscountingFactor * predictionEntropy + (1-entropyDiscountingFactor) * entropyThreshold;
				}
			}
		}
		
		if (temporalPooler != null){
			SimpleMatrix inputToSequencer = useBiasedInputInSequencer ? biasedSpatialFFOutputMatrix : spatialFFOutputMatrix;
			double[] spatialFFOutputDataVector;
			spatialFFOutputDataVector = inputToSequencer.getMatrix().data;	
			SimpleMatrix temporalFFInputVector = new SimpleMatrix(1, spatialFFOutputDataVector.length);
			temporalFFInputVector.getMatrix().data = spatialFFOutputDataVector;
			ffOutput = temporalPooler.feedForward(temporalFFInputVector);
			//ffOutput = sequencer.feedForward(temporalFFInputVector, needHelp);
		} else {
			ffOutput = biasSpatialFFOutput ? biasedSpatialFFOutputMatrix : spatialFFOutputMatrix;
		}
		
		if (stepsSinceSequenceStart <= markovOrder) temporalProbabilityMatrixToSend = new SimpleMatrix(ffOutput);		
		stepsSinceSequenceStart++;
		
		return temporalProbabilityMatrixToSend;
	}
	
	/**
	 * 
	 * @param inputMatrix
	 * @param correlationMatrix
	 * @return
	 */
	public SimpleMatrix feedBackward(SimpleMatrix inputMatrix){
		//Test input
		if (inputMatrix.numCols() != ffOutputMapSize || inputMatrix.numRows() != ffOutputMapSize) throw new IllegalArgumentException("The feed back input to the neocortical unit has to be a " + ffOutputMapSize + " x " + ffOutputMapSize + " matrix");

		fbInput = inputMatrix;
		
		if (needHelp){
			//Normalize
			SimpleMatrix normalizedInput = normalize(inputMatrix);
			
			if (temporalPooler != null){
				//Selection of best temporal model
				SimpleMatrix sequencerFBOutput = new SimpleMatrix(temporalPooler.feedBackward(normalizedInput));
				
				//Normalize
				SimpleMatrix normalizedSequencerFBOutput = normalize(sequencerFBOutput);
				
				//Transformation into matrix
				if (!noSpatial) normalizedSequencerFBOutput.reshape(spatialMapSize, spatialMapSize); 
				
				//Combine FB output from temporal pooler with bias and prediction (if enabled)
				biasMatrix = normalizedSequencerFBOutput;
			} else {
				biasMatrix = inputMatrix;
			}
			
			if (predictor != null) {
				biasMatrix = biasMatrix.elementMult(predictionMatrix);			
				
				biasMatrix = normalize(biasMatrix);			
			}
			stepsSinceSequenceStart = 0;
		} else {
			biasMatrix = predictionMatrix;
		}
		
		//biasMatrix = biasMatrix.plus(0.1 / biasMatrix.getNumElements()); //Add small uniform mass
		
		SimpleMatrix biasedTemporalFBOutput = biasMatrix;
		
		if (decider != null) chosenAction = chooseAction(biasedTemporalFBOutput);
		
		//Selection of best spatial model
		SimpleMatrix spatialPoolerFBOutputVector = biasedTemporalFBOutput;
		if (!noSpatial) spatialPoolerFBOutputVector = spatialPooler.feedBackward(biasedTemporalFBOutput);
		
		fbOutput = spatialPoolerFBOutputVector;
		
		return fbOutput;
	}
	
	private int chooseAction(SimpleMatrix state){
		int maxProbableState = -1;
		double maxProb = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < state.getNumElements(); i++){
			double d = state.get(i);
			if (d > maxProb){
				maxProb = d;
				maxProbableState = i;
			}
		}
		int action = decider.feedBack(maxProbableState, state);
		return action;
	}
	
	private SpatialPooler instantiateSpatialPooler(Random rand, int inputLength, int mapSize, double initialLearningRate, double stddev, double activationCodingFactor){
		SpatialPooler s = new SpatialPooler(rand, inputLength, mapSize, initialLearningRate, stddev, activationCodingFactor);
		return s;
	}
	
	private ActionDecider_Q instantiateActionDecider(int numPossibleActions, int numPossibleStates, double decayFactor, boolean offlineLearning, boolean useReactionaryDecider){
		ActionDecider_Q a;
		if (useReactionaryDecider){
			a = new ActionDecider_Q_Reactionary(numPossibleActions, numPossibleStates, decayFactor, offlineLearning);//TODO: Change parameters. Especially decay
		} else {
			a = new ActionDecider_Q(numPossibleActions, numPossibleStates, decayFactor, offlineLearning);//TODO: Change parameters. Especially decay
		}

		return a;
	}
	
	private Predictor_VOMM instantiatePredictor(int markovOrder, double learningRate, Random rand){
		Predictor_VOMM p = new Predictor_VOMM(markovOrder, learningRate, rand);
		return p;
	}
	
	private TemporalPooler instantiateTemporalPooler(Random rand, int inputLength, int mapSize, double initialLearningRate, double stddev, double activationCodingFactor, double decay){
		TemporalPooler p = new TemporalPooler(rand, inputLength, mapSize, initialLearningRate, stddev, activationCodingFactor, decay);
		return p;
	}
	
	private Sequencer instantiateSequencer(int markovOrder, int temporalGroupMapSize, int inputLength){
		Sequencer s = new Sequencer(markovOrder, temporalGroupMapSize, inputLength);
		return s;
	}
	
	private void feedDecider(SimpleMatrix inputToDecider, int actionPerformed, double reward){
		int maxProbableState = -1;
		double maxProb = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < inputToDecider.getNumElements(); i++){
			double d = inputToDecider.get(i);
			if (d > maxProb){
				maxProb = d;
				maxProbableState = i;
			}
		}
		decider.feedForward(maxProbableState, actionPerformed, reward, inputToDecider);
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
	
	public void resetActivity(){
		needHelp = false;
		active = false;
		chosenAction = -1;
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
	 * Bias the matrixToBias by element multiplying it with the biasMatrix.
	 * No changes are made to the original matrix.
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
		if (predictor != null) predictor.flush();
		if (temporalPooler != null) temporalPooler.flushTemporalMemory();
	}
	
	public void setLearning(boolean learning){
		if (spatialPooler != null) spatialPooler.setLearning(learning);
		if (decider != null) decider.setLearning(learning);
		if (predictor != null) predictor.setLearning(learning);
		if (temporalPooler != null) temporalPooler.setLearning(learning);
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
		if (!noSpatial) spatialPooler.sensitize(iteration);
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
		if (!noSpatial) return spatialPooler.getSOM();
		return null;
	}

	public void printPredictionModel() {
		if (predictor != null) predictor.printModel();
		
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
	
	public TemporalPooler getTemporalPooler(){ return temporalPooler;}

	/**
	 * @param entropyThresholdFrozen the entropyThresholdFrozen to set
	 */
	public void setEntropyThresholdFrozen(boolean entropyThresholdFrozen) {
		this.entropyThresholdFrozen = entropyThresholdFrozen;
	}
	
	public int getFeedForwardMapSize(){
		return ffOutputMapSize;
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
	
	public void printCorrelationMatrix(){
		decider.printQMatrix();
	}
	
	public Predictor_VOMM getPredictor(){
		return predictor;
	}
	
	public int getNextAction(){
		return chosenAction;
	}
	
	public int getMarkovOrder(){
		return markovOrder;
	}
	
	public void newEpisode(){
		decider.newEpisode();
		stepsSinceSequenceStart = 0;
	}
	
	public ActionDecider_Q getDecider(){
		return this.decider;
	}

	

}
