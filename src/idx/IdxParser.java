/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package idx;

import java.io.IOException;
import java.io.InputStream;

/**
 *
 * @author sam
 */
public class IdxParser {
    
    private InputStream in;
    private int numberLength;
    private boolean signed, floating;
    private NDArray data;
    
    // load the metadata and instantiate the array
    public IdxParser(InputStream input) throws IOException{
        in = input;
        if(in.read() != 0) error();
        if(in.read() != 0) error();
        signed = true;
        floating = false;
        switch(in.read()){
            case 0x08: //unsigned byte
                numberLength = 1;
                signed = false;
                break;
            case 0x09: //signed byte
                numberLength = 1;
                break;
            case 0x0B: // short
                numberLength = 2;
                break;
            case 0x0C: // int
                numberLength = 4;
                break;
            case 0x0D: // float
                numberLength = 4;
                floating = true;
                break;
            case 0x0E: // double
                numberLength = 8;
                floating = true;
                break;
            default:
                error();
        }
        // the next byte is the number of dimensions
        int dimensions = in.read();
        int[] shape = new int[dimensions];
        // the next few bytes are 2-byte sizes for each dimension
        for(int i = 0; i < dimensions; i++){
            shape[i] = ((in.read() * 256 + in.read()) * 256 + in.read()) * 256 + in.read();
        }
        // instantiate data based on that information TODO: Handle floating point data
        switch(numberLength){
            case 1:
                data = new NDArray<Byte>(shape);
                break;
            case 2:
                data = new NDArray<Short>(shape);
                break;
            case 4:
                data = new NDArray<Integer>(shape);
                break;
            case 8:
                data = new NDArray<Long>(shape);
                break;
        }
        
        
    }
    
    // parses the file
    public NDArray parseData() throws IOException{
        byte[] buffer = new byte[numberLength];
        int totalLen = data.get1dLength();
        for(int i = 0; i < totalLen; i++){
            in.read(buffer);
            data.set1d(i, byteArrayToInt(buffer));
        }
        return data;
    }
    
    private void error(){
        System.out.println("Error, the file did not adhere to the specs");
        System.exit(1);
    }
    
    private int byteArrayToInt(byte[] b){
        int value = 0;
        for(int i = 0; i < b.length; i++){
            value <<= 8;
            value += (int) b[i] & 0xFF;
        }
        return value;
    }
    
}
