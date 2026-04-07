function criticNet = criticNet()
%A deep critic network for DDPG with multiple layers
%
%   criticNet = CRITICNET() returns a layer graph suitable for use in
%   deep reinforcement learning with a deterministic policy gradient (DDPG)
%   agent. The network evaluates the Q-value of a given state-action pair. 
%   It consists of input layers for both state and action, followed by 
%   several hidden layers and an output layer representing the Q-value.
%
%   Example:
%       criticNet = CriticNet();
%       analyzeNetwork(criticNet);  % Visualize the network architecture
%
%   Inputs:
%       None
%
%   Outputs:
%       criticNet - Layer graph representing the critic network
%
%   Description:
%       This function builds a deep neural network with the following layers:
%       1. State Input Layer: A feature input layer representing the state.
%       2. Action Input Layer: A feature input layer representing the action.
%       3. Hidden Layers: Several fully connected layers with ReLU activation 
%          and dropout for regularization.
%       4. Output Layer: A fully connected layer to output the Q-value.
%
%   See also: rlDDPGAgent, rlRepresentation, createActorNet

% Define layer parameters (state, action, hidden layers, and dropout)
stateInputDim = 1; % State input dimension (e.g., 1 for simple case)
actionInputDim = 1; % Action input dimension (e.g., 1 for continuous action)
outputDimension = 1; % Q-value output (1 scalar)
hiddenLayerSizes = [256, 128, 64, 32]; % Sizes of hidden layers
dropoutRate = 0.2; % Dropout rate to regularize the network

% Build the critic network
criticNet = buildCriticNetwork(stateInputDim, actionInputDim, outputDimension, hiddenLayerSizes, dropoutRate);

end

% Local function to build the deep critic network
function lgraph = buildCriticNetwork(stateDim, actionDim, outputDim, hiddenSizes, dropoutRate)
%BUILDCRITICNETWORK Constructs the critic network with specified layers
%
%   lgraph = BUILDCRITICNETWORK(stateDim, actionDim, outputDim, hiddenSizes, dropoutRate)
%   constructs a layer graph representing a deep critic network architecture.
%   This function defines the layers, connects them, and returns a layerGraph.
%
%   Inputs:
%       stateDim - Integer, the dimension of the state input
%       actionDim - Integer, the dimension of the action input
%       outputDim - Integer, the dimension of the output (Q-value)
%       hiddenSizes - Array of integers, the number of neurons in each hidden layer
%       dropoutRate - Scalar, the rate of dropout regularization
%
%   Outputs:
%       lgraph - A LayerGraph object containing the critic network

% Initialize layers for state and action inputs
stateInput = featureInputLayer(stateDim, 'Normalization', 'none', 'Name', 'stateInput');
actionInput = featureInputLayer(actionDim, 'Normalization', 'none', 'Name', 'actionInput');

% Process the state input
stateLayers = [
    fullyConnectedLayer(hiddenSizes(1), 'WeightsInitializer', 'he', 'Name', 'state_fc1')
    reluLayer('Name', 'state_relu1')
    dropoutLayer(dropoutRate, 'Name', 'state_dropout1')
    ];

% Process the action input
actionLayers = [
    fullyConnectedLayer(hiddenSizes(1), 'WeightsInitializer', 'he', 'Name', 'action_fc1')
    reluLayer('Name', 'action_relu1')
    dropoutLayer(dropoutRate, 'Name', 'action_dropout1')
    ];

% Combine state and action layers
combinedLayers = [
    concatenationLayer(1, 2, 'Name', 'state_action_concat')
    fullyConnectedLayer(hiddenSizes(2), 'WeightsInitializer', 'he', 'Name', 'combined_fc1')
    reluLayer('Name', 'combined_relu1')
    dropoutLayer(dropoutRate, 'Name', 'combined_dropout1')
    ];

% Additional hidden layers for deeper learning
for i = 3:numel(hiddenSizes)
    combinedLayers = [
        combinedLayers
        fullyConnectedLayer(hiddenSizes(i), 'WeightsInitializer', 'he', 'Name', ['combined_fc' num2str(i)])
        reluLayer('Name', ['combined_relu' num2str(i)])
        dropoutLayer(dropoutRate, 'Name', ['combined_dropout' num2str(i)])
    ];
end

% Final output layer to predict the Q-value
outputLayer = [
    fullyConnectedLayer(outputDim, 'Name', 'output_fc')
    ];

% Assemble the layers into a LayerGraph
lgraph = layerGraph([
    stateInput
    actionInput
    stateLayers
    actionLayers
    combinedLayers
    outputLayer
    ]);

% Connect the layers (State + Action → Combined → Output)
lgraph = connectLayers(lgraph, 'state_fc1', 'state_action_concat/in1');
lgraph = connectLayers(lgraph, 'action_fc1', 'state_action_concat/in2');
end
