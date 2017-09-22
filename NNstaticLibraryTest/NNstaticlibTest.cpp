// NeuralNetwork.cpp : Defines the entry point for the console application.
//

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm> 
#include <algorithm>
#include "NNstaticlibTest.h"




namespace Neural {
	double Neuron::eta = 0.01;    // overall net learning rate 
	double Neuron::alpha = 0.01;   // momentum, multiplier of last deltaWeight 

	void printdata(std::vector<row> data) {

		for (unsigned i = 0; i < data.size(); i++) {

			for (unsigned j = 0; j < data[i].size(); j++) {

				std::cout << data[i][j] << " ";
			}

			std::cout << std::endl;
		}
	}

	std::vector<row> loadFromtext(std::string filename) {
		std::ifstream infile(filename);
		std::string line;
		std::vector<row> data;
		while (std::getline(infile, line)) {
			std::istringstream iss(line);
			row v;
			int i;
			while (iss >> i) {
				v.push_back(i);
			}
			data.push_back(v);
		}

		return data;
	}

	void splitDataIntoValuesAndLabels(std::vector<row>& x, std::vector<row>& y, std::vector<row> data) {

		for (unsigned i = 0; i < data.size(); i++) {
			row line_x = { data[i].at(0), data[i].at(1) };
			row line_y = { data[i].at(2) };

			x.push_back(line_x);
			y.push_back(line_y);
		}

	}



	/*
	============================================================================================
	Implementation of class Neuron

	============================================================================================
	*/

	Neuron::Neuron(unsigned numOutputs, unsigned myIndex, unsigned myActivationFlag)
	{

		for (unsigned c = 0; c < numOutputs; ++c)
		{
			m_outputWeights.push_back(Connection());
			m_outputWeights.back().weight = randomWeight(); // TODO
		}

		m_myIndex = myIndex;  // set index
		m_myActivationFlag = myActivationFlag;  //set activation flag
	}

	void Neuron::feedForward(const Layer & prevLayer)
	{
		double sum = 0.0;

		// Sum the previous layer's outputs (which are our inputs)
		// Include the bias node from the previous layer.

		for (unsigned n = 0; n < prevLayer.size(); ++n) {
			sum += prevLayer[n].getOutputVal() *
				prevLayer[n].m_outputWeights[m_myIndex].weight;
		}

		m_inputToNeuron = sum;

		m_outputVal = Neuron::transferFunction(sum, m_myActivationFlag);
	}

	void Neuron::calcOutputGradients(double targetVal)
	{
		double delta = targetVal - m_outputVal;
		m_gradient = delta * Neuron::transferFunctionDerivative(m_inputToNeuron, m_myActivationFlag);
	}

	void Neuron::calcHiddenGradients(const Layer & nextLayer)
	{
		double dow = sumDOW(nextLayer);
		m_gradient = dow * Neuron::transferFunctionDerivative(m_inputToNeuron, m_myActivationFlag);
	}

	void Neuron::updateInputWeights(Layer & prevLayer)
	{
		// The weights to be updated are in the Connection container
		// in the neurons in the preceding layer

		for (unsigned n = 0; n < prevLayer.size(); ++n) {
			Neuron &neuron = prevLayer[n];
			double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

			double newDeltaWeight =
				// Individual input, magnified by the gradient and train rate:
				eta
				* neuron.getOutputVal()
				* m_gradient
				// Also add momentum = a fraction of the previous delta weight;
				+ alpha
				* oldDeltaWeight;

			neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
			neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
		}
	}

	double Neuron::getweights(unsigned connectNum)
	{
		return m_outputWeights[connectNum].weight;
	}

	void Neuron::setWeights(unsigned connectNum, double weightVal)
	{
		m_outputWeights[connectNum].weight = weightVal;
	}

	double Neuron::sumDOW(const Layer & nextLayer) const
	{
		double sum = 0.0;

		// Sum our contributions of the errors at the nodes we feed.

		for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
			sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
		}

		return sum;
	}

	double Neuron::randomWeight(void) {
		return rand() / double(RAND_MAX);
	}

	double Neuron::transferFunction(double x, unsigned myActivationFlag)
	{
		double result;
		switch (myActivationFlag) {
		case(0): //linear
			result = x;
			break;
		case(1): //Relu
			result = std::max(0.001*x, x);
			break;
		case(2): //Tan
			result = tanh(x);
			break;
		}
		//TODO change to result
		return result;
	}

	double Neuron::transferFunctionDerivative(double x, unsigned myActivationFlag)
	{
		// input x is the input to the current neuron
		double result;
		switch (myActivationFlag) {
		case(0): //linear
			result = 1;
			break;
		case(1): //Relu
			result = (x > 0 ? 1 : 0.001);
			break;
		case(2): //Tan
			result = 1 - tanh(x)*tanh(x);
			break;
		}

		return result;
	}

	/*
	===============================================================================================
	Implementation of class Neural Network

	===============================================================================================
	*/

	void NeuralNet::Buildnetwork(const DoubleVector& LayerSizes, 
		StringVector& activations,
		double LearnRate,
		double Momentum)
	{
		// Set the values for learning rate and momentum:
		SetLearnRateAndMomentum(LearnRate, Momentum);

		unsigned numLayers = LayerSizes.size();  // Get number of layers in network

												 // for every layer add the correct number of neurons (plus bias)
		for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
			m_layers.push_back(Layer());	// add empty container to hold neruons of current layer

											// Get number of neurons in the next layer. If final layer set this to zero.
			unsigned numOutputs = layerNum == LayerSizes.size() - 1 ? 0 : LayerSizes[layerNum + 1];

			// Get activation flag for layer:
			std::string activationName = activations[layerNum];
			unsigned myActivationFlag = createActivationFlag(layerNum, activationName);

			// We have a new layer, now fill it with neurons, and add a bias neuron in each layer.
			for (unsigned neuronNum = 0; neuronNum <= LayerSizes[layerNum]; ++neuronNum) {
				m_layers.back().push_back(Neuron(numOutputs, neuronNum, myActivationFlag));
				std::cout << "Made a Neuron!" << std::endl;
			}

			// Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
			m_layers.back().back().setOutputVal(1.0);
		}
	}

	unsigned NeuralNet::createActivationFlag(unsigned layerNum, std::string & activation)
	{
		unsigned activationFlag;
		// The input layer the neurons should have no activations (linear)
		if (layerNum == 0) {
			activationFlag = 0;
		}
		else {
			if (activation == "Linear") {
				activationFlag = 0;
			}
			else if (activation == "Relu") {
				activationFlag = 1;
			}
			else if (activation == "Tan") {
				activationFlag = 2;
			}
			else {
				std::cout << "Activation Fucntion Incorrect for layer: " << layerNum << std::endl;
			}
		}

		return activationFlag;
	}

	void NeuralNet::feedForward(const std::vector<double>& inputVals)
	{
		assert(inputVals.size() == m_layers[0].size() - 1);

		// Assign (latch) the input values into the input neurons
		for (unsigned i = 0; i < inputVals.size(); ++i) {
			m_layers[0][i].setOutputVal(inputVals[i]);
		}

		// forward propagate
		for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
			Layer &prevLayer = m_layers[layerNum - 1];
			for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
				m_layers[layerNum][n].feedForward(prevLayer);
			}
		}

	}

	void NeuralNet::backProp(const std::vector<double>& targetVals)
	{
		// --- Calculate overall net error (RMS of output neuron errors) ---

		Layer &outputLayer = m_layers.back();   // get reference to final layer
		m_error = 0.0;

		// find the error between the target values and the outputs:
		for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
			double delta = targetVals[n] - outputLayer[n].getOutputVal();
			m_error += delta * delta;
		}
		m_error /= outputLayer.size() - 1; // get average error squared
		m_error = sqrt(m_error); // RMS

								 // --------

								 // Calculate output layer gradients

		for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
			outputLayer[n].calcOutputGradients(targetVals[n]);
		}

		// Calculate hidden layer gradients

		for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
			Layer &hiddenLayer = m_layers[layerNum];
			Layer &nextLayer = m_layers[layerNum + 1];

			for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
				hiddenLayer[n].calcHiddenGradients(nextLayer);
			}
		}

		// For all layers from outputs to first hidden layer,
		// update connection weights

		for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
			Layer &layer = m_layers[layerNum];
			Layer &prevLayer = m_layers[layerNum - 1];

			for (unsigned n = 0; n < layer.size() - 1; ++n) {
				layer[n].updateInputWeights(prevLayer);
			}
		}
	}

	void NeuralNet::getResults(std::vector<double>& resultVals) const
	{
		resultVals.clear();

		for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
			resultVals.push_back(m_layers.back()[n].getOutputVal());
		}
	}

	void NeuralNet::SetLearnRateAndMomentum(double learn, double momentum)
	{
		Neuron::setLearningRate(learn);
		Neuron::setMomentum(momentum);
	}

	void NeuralNet::SaveNetwork(cString filename)
	{	
		//std::string filename = "C:\\random\\NetworkWeights.txt";
		std::ofstream myfile;
		myfile.open(filename);

		// For every layer in the network excluding the last
		for (unsigned layerNum = 0; layerNum < m_layers.size() - 1; layerNum++) {
			//Get size of next layer excluding the bias
			unsigned NextLayerSize = m_layers[layerNum + 1].size() - 1; // 

			// For every neuron in a layer
			for(unsigned NeuronNum = 0; NeuronNum < m_layers[layerNum].size(); NeuronNum++){
				double output = m_layers[layerNum][NeuronNum].getOutputVal();

				// For every connection from a neuron to a neuron in the next layer
				for (unsigned ConnectNum = 0; ConnectNum < NextLayerSize; ConnectNum++) {
					double weight = m_layers[layerNum][NeuronNum].getweights(ConnectNum);
					// Save weight to file
					myfile << weight << " ";
				}
				myfile << std::endl;  // line line seperates weights of neurons
			}
			
		}

		myfile.close();
	}


	void NeuralNet::LoadNetwork(cString filename)
	{	
		// Section of below gets weights from text files and stores them in a vector of vectors
		std::ifstream infile(filename);
		std::string line;
		std::vector<row> savedWeights;
		while (std::getline(infile, line)) {
			std::istringstream iss(line); // get a line of text file
			row v; // store a line of text
			double i; // store one weight
			while (iss >> i) {
				v.push_back(i);
			}
			savedWeights.push_back(v);
		}

		// Following code adds the weights to the network
		// For every layer in the network excluding the last
		unsigned NeuronCount = 0;
		for (unsigned layerNum = 0; layerNum < m_layers.size() - 1; layerNum++) {
			//Get size of next layer excluding the bias
			unsigned NextLayerSize = m_layers[layerNum + 1].size() - 1; // 

																		// For every neuron in a layer
			for (unsigned NeuronNum = 0; NeuronNum < m_layers[layerNum].size(); NeuronNum++) {
				double output = m_layers[layerNum][NeuronNum].getOutputVal();

				// For every connection from a neuron to a neuron in the next layer
				for (unsigned ConnectNum = 0; ConnectNum < NextLayerSize; ConnectNum++) {
					// set weight value to corresponding in the file

					double weights = savedWeights[NeuronCount][ConnectNum];

					m_layers[layerNum][NeuronNum].setWeights(ConnectNum, weights);
					// Save weight to file
				}

				++NeuronCount;
			}
		}


	}


	void helper::getArgMax(DoubleVector & myVec, int & argMax)
	{
		argMax = std::distance(myVec.begin(), std::max_element(myVec.begin(), myVec.end()));
	}

	void helper::getMax(DoubleVector & myVec, double & Max)
	{
		Max = *std::max_element(myVec.begin(), myVec.end());
	}

}
