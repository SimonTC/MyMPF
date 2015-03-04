package stcl.test.predictors;

import java.util.LinkedList;

import stcl.algo.predictors.VOMM;
import stcl.algo.util.trie.Trie;

public class VOMMTest {

	
	public static void main(String[] args) {
		String data = "aaaaaccbbabcbba";
		VOMMTest t = new VOMMTest();
		t.runActive(4, data);

	}
	
	public void runActive(int maxDepth, String data){
		VOMM<Character> predictor = new VOMM<Character>(maxDepth, 1);
		Character prediction = 'ø';
				
		for (int iteration = 1; iteration < maxDepth * 10; iteration++){
			int error = 0;
			for (int i = 0; i < data.length(); i++){
				Character v = data.charAt(i);
				if (prediction == null){
					error++;
				} else{
					if (!prediction.equals(v)) error++;
				}
				//System.out.println("predicted: " + prediction + " Actual: " + v);
				predictor.addSymbol(v);
				
				//predictor.printTrie();
				
				prediction = predictor.predict();
				
			}
			double avgError = (double) error / data.length();
			System.out.println("Average error after iteration " + iteration + ": " + avgError);
		}
	
	}
	
	

}
