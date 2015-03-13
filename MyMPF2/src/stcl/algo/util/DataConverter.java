package stcl.algo.util;

public class DataConverter {
	
	/**
	 * 
	 * @param i integer to convert to bitstring
	 * @param length length of bit string. 0's will be added in front of the bit string if it is not long enough
	 * @return
	 */
	public static String intToBitString(int integer, int length){
		String s = Integer.toBinaryString(integer);
		while (s.length() < length){
			s = "0" + s;
		}
		
		return s;
	}
	
	public static double[] intToBitStringDouble(int integer, int length){
		String bitString = intToBitString(integer, length);
		double[] arr = new double[length];
		for (int i = 0; i < bitString.length(); i++){
			arr[i] = Double.parseDouble(bitString.substring(i, i+1));
		}
		return arr;
	}
	
	/**
	 * 
	 * @param s bistring to convert to int
	 * @return
	 */
	public static int bitstringToInt(String s){
		return Integer.parseInt(s, 2);
	}

}
