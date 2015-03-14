package stcl.algo.util;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;

public class FileWriter {
	
	private BufferedWriter writer;
	private File file;
	
	/**
	 * No args constructor used when the file to be written to can change during the life of the object
	 */
	public FileWriter() {
		// 
	}
	
	/**
	 * Constructor used when the same file is used for all writing
	 * @param fileName
	 */
	public FileWriter(String fileName) {
		file = new File(fileName);
	}
	
	public void openFile(String filename, boolean append) throws IOException{
		File file = new File(filename);
		writer = new BufferedWriter(new java.io.FileWriter(file, append));
	}
	
	/**
	 * Use if file is given in the constructor
	 * @param append
	 * @throws IOException
	 */
	public void openFile(boolean append) throws IOException{
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
