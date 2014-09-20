package stcl.algo.util;
/**
 * This class is used to implement the exponential decay function 
 * @author Simon
 *
 */
public class ExponentialDecayFunction {

	private double startValue;
	private double minValue;
	private double denominator;
	
	
	public ExponentialDecayFunction(double startValue, double minValue,  double denominator) {
		//TODO: change name of denominatorInDecayConstant
		this.startValue = startValue;
		this.minValue = minValue;
		this.denominator = denominator;
		
	}
	
	public double decayValue(int tick){
		double d = startValue * Math.exp(-tick/denominator);
		if (d < minValue) d = minValue;
		return d;
	}
	
	

}
