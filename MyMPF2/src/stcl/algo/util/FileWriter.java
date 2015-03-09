package stcl.algo.util;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;

public class FileWriter {
	
	private BufferedWriter writer;
	
	public void openFile(String filename, boolean append) throws IOException{
		File file = new File(filename);
		writer = new BufferedWriter(new java.io.FileWriter(file, append));
	}
	
	public void closeFile() throws IOException{
		writer.close();
	}
	
	public void writeLine(String line) throws IOException{
		write(line);
		writer.newLine();
	}
	
	public void write(String line) throws IOException{
		writer.write(line);
	}

}
