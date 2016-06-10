/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package idx;

import java.util.ArrayList;

/**
 * Simulates a N-dimensional array that allows assignment and access to elements
 * @author sam
 */
public class NDArray<T> {
    private ArrayList<T> data;
    private int[] shape;
    private int totalLen;
    public NDArray(int[] shape){
        totalLen = 1;
        for(int i = shape.length-1; i >= 0; i--) totalLen *= shape[i];
        data = new ArrayList<T>(totalLen);
        this.shape = shape;
    }
    
    public T get(int[] position) throws IndexOutOfBoundsException{
        if(position.length != shape.length) throw new IndexOutOfBoundsException();
        int pos = 0;
        int stepSize = data.size();
        for(int i = 0; i < position.length; i++){
            // divide the stepSize by shape[i]
            stepSize/= shape[i];
            // at each dimension, add position[i] * stepSize
            if(position[i] >= shape[i]) throw new IndexOutOfBoundsException();
            pos += position[i] * stepSize;
        }
        return data.get(pos);
    }
    public void set(int[] position, T value) throws IndexOutOfBoundsException{
        if(position.length != data.size()) throw new IndexOutOfBoundsException();
        int pos = 0;
        int stepSize = data.size();
        for(int i = 0; i < position.length; i++){
            // divide the stepSize by shape[i]
            stepSize/= shape[i];
            // at each dimension, add position[i] * stepSize
            if(position[i] >= shape[i]) throw new IndexOutOfBoundsException();
            pos += position[i] * stepSize;
        }
        data.set(pos, value);
    }
    
    public int[] getShape(){
        return shape;
    }
    public int get1dLength(){
        return totalLen;
    }
    
    // gets without dimensions - allows you to access the underlying structure
    public T get1d(int position){
        return data.get(position);
    }
    
    public void set1d(int position, T value){
        if(position < data.size()) data.set(position, value);
        else{
            while(data.size() < position) data.add(null);
            data.add(value);
        }
    }
    
    public String toString(){
        
        return "NDArray with " + shape.length + " dimensions:\n" + data.toString();
    }
}
