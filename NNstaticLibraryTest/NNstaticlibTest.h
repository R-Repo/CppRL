/*
=====================================================================================================
										Feedforward Neural Network
				Overview:
					Feedforward network is built using two class, neuron and NeuralNet. A network
					is defined by the number of layers, number of neuron in each layer and the activation
					function in each layer. The network is built in the class NerualNet and each neuron
					in the network is an instance of the class neuron. This network uses the square loss
					only.

				Activation Functions:
					The activation functions must be defined for every layer of the network (including
					input layer). This is done my passing a vector of strings to the NerualNet class.

					'Relu'

					'Tanh'

					'linear'

					'sigmoid'

					Example activations for a 4 layer network:
					std::vector<std::string> activations = { "Linear", "Relu", "Tan", "sigmoid" };

				Layer:
					The network must be passed a parameter LayerSizes. This defined the number of layers and
					the number of neurons in each layer.

					Example for network with four layers with 30 neurons in input, 2 hidden layers and 10
					neurons in the output layer:
					std::vector<unsigned> Layersizes = {30, 126, 256, 10};

=======================================================================================================
*/

#pragma once
#include <vector>

namespace Neural {

	typedef std::vector<double> DoubleVector;		// needed to use vectors in UE4;
	typedef std::vector<std::string> StringVector; // ^
	typedef std::vector<double> row;
	typedef std::vector<row> VecOfVecDoubles;
	typedef std::string cString;

	struct MaxActions
	{
		int argMax;
		double max;
	};

	// Each neuron has connection to all neurons in the next layer.
	struct Connection
	{
		double weight;			// weight between neruon and a single neuron in next layer
		double deltaWeight;     // the derviative being back propogated
	};

	class Neuron;

	typedef std::vector<Neuron> Layer;  // Layer of the neural network

// ====================================================================================
//						 ************** CLass Neuron *****************

// The neuron is a single node within the neural network. Every node within a the
// neural network will be defined by an instance of this class. Each neuron is defined
// by the number of connections (number of neurons in next layer) and the activation
// function.

// ====================================================================================
	class Neuron
	{
	public:
		Neuron(unsigned numOutputs, unsigned myIndex,				// Initialise a neuron:
			unsigned myActivationFlag);

		void setOutputVal(double val) { m_outputVal = val; }		// Set the output value of a neuron.
		void feedForward(const Layer &prevLayer);					// Forward propogate values within the network
		double getOutputVal(void) const { return m_outputVal; }		// get output value (after activation) of neuron
		void calcOutputGradients(double targetVal);					// gradients of final layer (Square loss)
		void calcHiddenGradients(const Layer &nextLayer);			// 
		void updateInputWeights(Layer &prevLayer);					// Update weights
		static void setLearningRate(double learningrate) { eta = learningrate; };
		static void setMomentum(double Momentum) { alpha = Momentum; };
		double getweights(unsigned connectNum);
		void setWeights(unsigned connectNum, double weightVal);



	private:
		static double randomWeight(void);					// Create a random weight when neuron is intialised
		static double transferFunction(double x,			// MUST UPDATE IF ADDING NEW ACTIVATION FUNCTION
			unsigned myActivationFlag);
		static double transferFunctionDerivative(double x,	// MUST UPDATE IF ADDING NEW ACTIVATIONFUNCTION
			unsigned myActivationFlag);
		double sumDOW(const Layer &nextLayer) const;

		std::vector<Connection> m_outputWeights;		// connection to every neuron in next layer
		unsigned m_myIndex;								// Index of neuron in it's layer
		unsigned m_myActivationFlag;					// Defines what type of activation function to use
		double m_outputVal;								// Output value of the neuron being feed forward.
		double m_inputToNeuron;							// sum (previous layer activations * weights) + bias
		double m_gradient;								// gradient which is passed back

														// Learning parameters
		static double eta;								// Learning rate
		static double alpha;							// Momentum
	};


	class NeuralNet
	{
	public:

		void Buildnetwork(const DoubleVector &LayerSizes,			// Initialse a neural network. 
			StringVector &activations,
			double LearnRate,
			double Momentum);

		unsigned createActivationFlag(unsigned layerNum,			// turns string inputs for the activation flag 
			std::string &activation);								//into numbers

		void feedForward(const DoubleVector &inputVals);		// Forward propogate the inputs for the neural network

		void backProp(const DoubleVector &targetVals);		// BackPropogate the errors

		void getResults(DoubleVector &resultVals) const;		// Get outputs of the final layer. Have to give a vector which it will change
		void SetLearnRateAndMomentum(double learn, double momentum);
		
		void SaveNetwork(cString filename);
		void LoadNetwork(cString filename);

	private:
		std::vector<Layer> m_layers;				// vector containing vectors of neurons. Each input is one layer
		double m_error;                             // mean square error of the network		void SetLearnRateAndMomentum(double learn, double momentum);  
	};


	class helper
	{
	public:
		static void getArgMax(DoubleVector &myVec, int &argMax);
		static void getMax(DoubleVector &myVec, double &Max);

	};

}