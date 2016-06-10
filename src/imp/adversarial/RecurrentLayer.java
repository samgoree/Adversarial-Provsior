/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package imp.adversarial;

import imp.neuralnet.ActivationFunction;
import imp.neuralnet.Layer;
import imp.neuralnet.Neuron;
import imp.neuralnet.Source;

/**
 * A Recurrent neural network layer that stores its prior values and uses them
 * as input alongside the actual inputs from the previous layer
 * @author SG
 * 6/9/16
 */
public class RecurrentLayer extends Layer{
    
    private Double[] priorValues; // these are saved whenever fire or use is called
    private boolean activationInProgress;
    private int numberInLayer;
    
    /**
     * Creates a new RecurrentLayer
     * @param layerIndex the index
     * @param numberInLayer the number of neurons in the layer
     * @param type the activation function of the layer
     * @param numberInPrevLayer the number of nodes in the previous layer
     */
    public RecurrentLayer(int layerIndex, int numberInLayer, ActivationFunction type, int numberInPrevLayer){
        super(layerIndex, numberInLayer, type, numberInPrevLayer + numberInLayer);
        this.numberInLayer = numberInLayer;
        activationInProgress = false;
    }
    
    /*
     * Fire all the Neurons in this layer, based on the Source AND this layer's 
     * previous values, with this layer first, setting the output value 
     * and derivative evaluated at the net value.
     */
    public void fire(Source source)
    {
        activationInProgress = true;
        priorValues = getOutput();
        super.fire(new RecurrentSource(this, numberInLayer, source));
        activationInProgress = false;
    }
    
    /*
     * Use all the Neurons in this layer, based on the Source AND this layer's 
     * previous values, with this layer first, setting the output value 
     * at the net value.
     */
    public void use(Source source)
    {
        activationInProgress = true;
        priorValues = getOutput();
        super.use(new RecurrentSource(this, numberInLayer, source));
        activationInProgress = false;
    }
    
    public double get(int i)
    {
        if(activationInProgress) return priorValues[i];
        else return super.get(i);
    }
    
        /*
     * Adjust the weights on each neuron in this layer
     */
    public void adjustWeights(Source source, double rate)
    {
        activationInProgress = true; // not actually, but it makes it use the prior values
        super.adjustWeights(new RecurrentSource(this, numberInLayer, source), rate);
        activationInProgress = false;
    }
    
    // Allows you to pass two sources into a neuron as one source
    private class RecurrentSource implements Source{
        Source s1,s2;
        int s1Size;
        private RecurrentSource (Source s1, int s1Size, Source s2){
            this.s1 = s1;
            this.s2 = s2;
            this.s1Size = s1Size;
        }
        
        public double get(int j){
            if(j < s1Size)
                return s1.get(j);
            else
                return s2.get(j-s1Size);
        }
    }
}
