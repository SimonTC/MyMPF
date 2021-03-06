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
/**
 * The neocortical unit is the main computational unit in the network.
 * It consists of four sub elements: a spatial pooler, a temporal pooler, a predictor and a action decider.
 * @author Simon
 *
 */
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
	private int ffOutputVectorLength;
	private int ffOutputMapSize;
	
	private boolean needHelp;
	
	//Entropy related
	private double predictionEntropy;
	private double entropyThreshold;
	private double entropyDiscountFactor;
	private double episodeEntropy;
	private boolean entropyThresholdFrozen;
	
	private boolean usePrediction;
	private boolean active;
	
	private int episodeLength;

	
	
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
	 * @param ffInputLength
	 * @param spatialMapSize
	 * @param temporalMapSize
	 * @param markovOrder
	 * @param numPossibleActions
	 * @param usePrediction
	 * @param reactionary
	 * @param offlineLearning
	 */
	public NeoCorticalUnit(int ffInputLength, int spatialMapSize, int temporalMapSize, int markovOrder, int numPossibleActions, boolean usePrediction, boolean reactionary, boolean offlineLearning) {
		initialize(ffInputLength, spatialMapSize, temporalMapSize, markovOrder, numPossibleActions, usePrediction, reactionary, offlineLearning, null);
	}
	
	/**
	 * 
	 * @param ffInputLength
	 * @param spatialMapSize
	 * @param temporalMapSize
	 * @param markovOrder
	 * @param numPossibleActions
	 * @param usePrediction
	 * @param reactionary
	 * @param offlineLearning
	 * @param rand
	 */
	public NeoCorticalUnit(int ffInputLength, int spatialMapSize, int temporalMapSize, int markovOrder, int numPossibleActions, boolean usePrediction, boolean reactionary, boolean offlineLearning, Random rand) {
		initialize(ffInputLength, spatialMapSize, temporalMapSize, markovOrder, numPossibleActions, usePrediction, reactionary, offlineLearning, rand);
	}
	
	/**
	 * Create the Neocortical unit from the initialization string created by the toInitializationString() method.
	 * All elements of the Unit are initialized with the values in the initialization string
	 * @param initializationString
	 */
	public NeoCorticalUnit(String initializationString){
		String[] lines = initializationString.split(SomConstants.LINE_SEPARATOR);
		String[] unitInfo = lines[0].split(" ");
		initialize(Integer.parseInt(unitInfo[0]), Integer.parseInt(unitInfo[1]), Integer.parseInt(unitInfo[2]), Integer.parseInt(unitInfo[3]), Integer.parseInt(unitInfo[4]), Boolean.parseBoolean(unitInfo[5]), Boolean.parseBoolean(unitInfo[6]), Boolean.parseBoolean(unitInfo[7]), null);
		if (!noSpatial) spatialPooler = new SpatialPooler(initializationString, 1);
		if (!noTemporal){
			int temporalStart = 0;
			String[] tmp = initializationString.split(SomConstants.LINE_SEPARATOR);
			for (int i = 0; i < tmp.length; i++){
				if (tmp[i].equalsIgnoreCase("TEMPORAL")){
					temporalStart = i + 1;
					break;
				}
			}
			temporalPooler = new TemporalPooler(initializationString, temporalStart);
		}
	}
	public String toInitializationString(){
		String s = ffInputVectorSize + " " + spatialMapSize + " " + temporalMapSize + " " + markovOrder + " " + numPossibleActions + " " + usePrediction + " " + reactionary + " " + offlineLearning + SomConstants.LINE_SEPARATOR;
		if (!noSpatial) s+= spatialPooler.toInitializationString();
		if (!noTemporal) s+= "TEMPORAL" + SomConstants.LINE_SEPARATOR +  temporalPooler.toInitializationString();
		return s;
	}
	
	private void initialize(int ffInputLength, int spatialMapSize, int temporalMapSize, int markovOrder, int numPossibleActions, boolean usePrediction, boolean reactionary, boolean offlineLearning, Random rand){
		//Test arguments
		if (ffInputLength < 1) throw new IllegalArgumentException("Input length has to be greater than 0");
		
		//Instantiate sub-components
		int spatialOutputLength = ffInputLength;
		noSpatial = spatialMapSize < 1;
		if (!noSpatial) {
			spatialPooler = instantiateSpatialPooler(ffInputLength, spatialMapSize, 0.1, Math.sqrt(spatialMapSize), 0.125, rand);
			spatialOutputLength = (int) Math.pow(spatialMapSize, 2);
		}
		
		if (numPossibleActions > 0) decider = instantiateActionDecider(numPossibleActions, spatialOutputLength, 0.9, offlineLearning, reactionary);
		
		if (markovOrder > 0) predictor = instantiatePredictor(markovOrder, 0.1); 
		
		double decay = calculateDecay(markovOrder,0.01);

		noTemporal = (temporalMapSize < 1);
		if (temporalMapSize > 0) temporalPooler = instantiateTemporalPooler(spatialOutputLength, temporalMapSize, 0.1, Math.sqrt(temporalMapSize), 0.125, decay, rand);
		
		//Set map and output sizes
		this.spatialMapSize = spatialMapSize;
		this.temporalMapSize = temporalMapSize;
		
		ffOutputVectorLength = (int) (noTemporal ? Math.pow(spatialMapSize,2) : Math.pow(temporalMapSize,2));
		
		ffOutputMapSize = noTemporal ? spatialMapSize : temporalMapSize;

		ffInputVectorSize = ffInputLength;
		
		
		
		//Set flags
		this.usePrediction = usePrediction;
		needHelp = false;
		entropyThresholdFrozen = false;
		biasSpatialFFOutput = false; 
		useBiasedInputInPredictor = false;
		useBiasedInputInSequencer = false;
		
		//Set fields
		
		entropyDiscountFactor = 0.1;
		entropyThreshold = 0;
		this.markovOrder = markovOrder;
		
		this.numPossibleActions = numPossibleActions; 
		this.reactionary = reactionary;
		this.offlineLearning = offlineLearning;
		this.temporalMapSize = temporalMapSize;
				
		//Initialize matrices
		if (noSpatial){
			biasMatrix = new SimpleMatrix(1, ffInputLength);
			predictionMatrix = new SimpleMatrix(1, ffInputLength);
		} else {
			biasMatrix = new SimpleMatrix(spatialMapSize, spatialMapSize);
			predictionMatrix = new SimpleMatrix(spatialMapSize, spatialMapSize);
		}
		
		this.newEpisode();
	}
	
	/**
	 * Resets all temporal knowledge of the elements in the unit.
	 * the entropy threshold is not reset as it makes sense for the unit to retain knowledge about the prediction difficulty of the world.
	 */
	public void newEpisode(){
		if (decider!= null) decider.newEpisode();
		if (predictor != null) predictor.newEpisode();
		if (temporalPooler != null) temporalPooler.newEpisode();
		stepsSinceSequenceStart = 0;
		
		ffOutput = new SimpleMatrix(this.ffOutputMapSize, this.ffOutputMapSize);
		fbOutput = new SimpleMatrix(1, ffInputVectorSize);
		temporalProbabilityMatrixToSend = new SimpleMatrix(ffOutputMapSize, ffOutputMapSize);
		
		if (biasMatrix != null) biasMatrix.set(1);
		if (predictionMatrix != null) {
			predictionMatrix.set(1);
			predictionMatrix = Normalizer.normalize(predictionMatrix);
		}
		
		this.updateEntropyThreshold();
		episodeEntropy = 0;
		episodeLength = 0;
		stepsSinceSequenceStart = 0;
		this.resetActivity();
	}
	
	private void updateEntropyThreshold(){
		if (!entropyThresholdFrozen){
			if (episodeLength > 0){
				double avgEpisodeEntropy = episodeEntropy / (double) episodeLength;
				entropyThreshold = entropyDiscountFactor * avgEpisodeEntropy + (1-entropyDiscountFactor) * entropyThreshold;
			}
		}
	}
	
	public SimpleMatrix feedForward(SimpleMatrix inputVector){
		return this.feedForward(inputVector, 0,0);
	}

	public SimpleMatrix feedForward(SimpleMatrix inputVector, double reward, int actionPerformed){
		//Test input
		if (!inputVector.isVector()) throw new IllegalArgumentException("The feed forward input to the neocortical unit has to be a vector");
		if (inputVector.numCols() != ffInputVectorSize) throw new IllegalArgumentException("The feed forward input to the neocortical unit has to be a 1 x " + ffInputVectorSize + " vector");
		
		active = true;
		
		episodeLength++;
		
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
				episodeEntropy += predictionEntropy;
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
		int action = decider.feedBack(state);
		return action;
	}
	
	private SpatialPooler instantiateSpatialPooler(int inputLength, int mapSize, double initialLearningRate, double stddev, double activationCodingFactor, Random rand){
		SpatialPooler s;
		if (rand!= null){
			s = new SpatialPooler(inputLength, mapSize, initialLearningRate, stddev, activationCodingFactor, rand);			
		} else {
			s = new SpatialPooler(inputLength, mapSize, initialLearningRate, stddev, activationCodingFactor);
		}
		return s;
	}
	
	private ActionDecider_Q instantiateActionDecider(int numPossibleActions, int numPossibleStates, double decayFactor, boolean offlineLearning, boolean useReactionaryDecider){
		ActionDecider_Q a;
		if (useReactionaryDecider){
			a = new ActionDecider_Q_Reactionary(numPossibleActions, numPossibleStates, decayFactor, offlineLearning);
		} else {
			a = new ActionDecider_Q(numPossibleActions, numPossibleStates, decayFactor, offlineLearning);
		}

		return a;
	}
	
	private Predictor_VOMM instantiatePredictor(int markovOrder, double learningRate){
		Predictor_VOMM p = new Predictor_VOMM(markovOrder, learningRate);
		return p;
	}
	
	private TemporalPooler instantiateTemporalPooler(int inputLength, int mapSize, double initialLearningRate, double stddev, double activationCodingFactor, double decay, Random rand){
		TemporalPooler p;
		if (rand != null){
			p = new TemporalPooler(inputLength, mapSize, initialLearningRate, stddev, activationCodingFactor, decay, rand);			
		} else {
			p = new TemporalPooler(inputLength, mapSize, initialLearningRate, stddev, activationCodingFactor, decay);
		}
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
	
	public int getFeedForwardVectorLength(){
		return ffOutputVectorLength;
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
	

	
	public ActionDecider_Q getDecider(){
		return this.decider;
	}
	
	public void setBiasSpatialFFOutput(boolean flag){
		this.biasSpatialFFOutput = flag;
	}
	

}
