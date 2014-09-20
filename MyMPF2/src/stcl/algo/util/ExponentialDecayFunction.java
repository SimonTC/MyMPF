package stcl.algo.util;
/**
 * This class is used to implement the exponential decay function 
 * @author Simon
 *
 */
public class ExponentialDecayFunction {

	private double startValue;
	private double minValue;
	private double decayConstant;
	
	
	public ExponentialDecayFunction(double startValue, double minValue,int maxTicks,  double denominatorInDecayConstant) {
		//TODO: change name of denominatorInDecayConstant
		this.startValue = startValue;
		decayConstant = maxTicks / Math.log(denominatorInDecayConstant);
		
	}
	
	public double decayValue(int tick){
		double d = startValue * Math.exp(-tick/decayConstant);
		if (d < minValue) d = minValue;
		return d;
	}
	
	

}
