/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package adversarial.provisor;

import idx.IdxParser;
import idx.NDArray;
import imp.adversarial.RecurrentNetwork;
import imp.neuralnet.ActivationFunction;
import imp.neuralnet.Logsig;
import imp.neuralnet.Network;
import imp.neuralnet.Sample;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author sam
 */
public class AdversarialProvisor {
    
    static double etaPlus = 1.2; //for rprop
    static double etaMinus = 0.5;

    /**
     * This is just testing, sorry for spaghetti code!
     */
    public static void main(String[] args) {
        try {
            
            Logsig ls = new Logsig();
            int[] size = new int[]{100,100,10};
            boolean[] isRecurrent = new boolean[]{false, false, false};
            // Simple example to make sure the code works
            RecurrentNetwork rn = new RecurrentNetwork(28*28, size, isRecurrent, new ActivationFunction[]{ls, ls, ls});
            Network nn = new Network(3, size,
            new ActivationFunction[]{ls, ls, ls}, 2);
            // read in a dataset and train on it - here I'm using the mnist dataset
            IdxParser p = new IdxParser(new FileInputStream("/home/sam/Documents/mnist/train-images.idx3-ubyte"));
            NDArray<Byte> data = p.parseData();
            IdxParser q = new IdxParser(new FileInputStream("/home/sam/Documents/mnist/train-labels.idx1-ubyte"));
            NDArray<Byte> labels = q.parseData();
            Sample[] samples = new Sample[data.getShape()[0]];
            
            // Load the data into samples
            for(int i = 0; i < samples.length; i++){
                samples[i] = new Sample(data.getShape()[1] * data.getShape()[2], 10);
                for(int j = 0; j < data.getShape()[1]; j++){
                    for(int k = 0; k < data.getShape()[2]; k++){
                        samples[i].setInput(j * data.getShape()[2] + k, (int)data.get(new int[]{i,j,k}));
                    }
                }
                samples[i].setOutput(labels.get(new int[]{i}), 1);
            }
            // Train for 100 epochs
            for(int i = 0; i < 100; i++){
                if(i%10 == 0) System.out.println("Starting epoch " + i);
                for(int j = 0; j < samples.length; j++){
                    rn.fire(samples[j]);
                    rn.setSensitivity(samples[j]);
                    rn.accumulateGradient(samples[j]);
                    rn.adjustByRprop(etaPlus, etaMinus);
                    nn.fire(samples[j]);
                    nn.setSensitivity(samples[j]);
                    nn.accumulateGradient(samples[j]);
                    nn.adjustByRprop(etaPlus, etaMinus);
                }
            }
            
            //Test on the test data
            p = new IdxParser(new FileInputStream("/home/sam/Documents/mnist/t10k-images.idx3-ubyte"));
            data = p.parseData();
            q = new IdxParser(new FileInputStream("/home/sam/Documents/mnist/t10k-labels.idx1-ubyte"));
            labels = q.parseData();
            Sample[] testSamples = new Sample[data.getShape()[0]];
            
            // Load the data into samples
            for(int i = 0; i < testSamples.length; i++){
                testSamples[i] = new Sample(data.getShape()[1] * data.getShape()[2], 10);
                for(int j = 0; j < data.getShape()[1]; j++){
                    for(int k = 0; k < data.getShape()[2]; k++){
                        testSamples[i].setInput(j * data.getShape()[2] + k, data.get(new int[]{i,j,k}));
                    }
                }
                testSamples[i].setOutput(labels.get(new int[]{i}), 1);
            }
            int correct = 0;
            boolean fail;
            for(int i = 0; i < testSamples.length; i++){
                rn.use(testSamples[i]);
                fail = false;
                for(int j = 0; j < 10; j++){
                    if(rn.getOutput()[j] != testSamples[i].getOutput(j)) fail = true;
                }
                if(!fail) correct++;
            }
            System.out.println("Total Correct: " + correct + "/" + testSamples.length);
            System.out.println("Percentage: " + correct/testSamples.length * 100 + "%");
            
        } catch (IOException ex) {
            Logger.getLogger(AdversarialProvisor.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
}
