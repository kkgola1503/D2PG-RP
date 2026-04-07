function actorNet = actorNet()
% A deep actor network for DDPG with multiple layers
%
%   actorNet = ACTORNET() returns a layer graph suitable for use in
%   deep reinforcement learning with a deterministic policy gradient (DDPG)
%   agent. The network consists of several hidden layers, dropout regularization,
%   and an output layer bounded by tanh to constrain the action space between
%   [-1, 1]. This architecture is designed to handle continuous action spaces.
%
%   Example:
%       actorNet = ActorNet();
%       analyzeNetwork(actorNet);  % Visualize the network architecture
%
%   Inputs:
%       None
%
%   Outputs:
%       actorNet - Layer graph representing the actor network
%       
%   Description:
%       This function builds a deep neural network with the following layers:
%       1. Input Layer: A feature input layer representing the state.
%       2. Hidden Layers: Three fully connected layers with ReLU activation and
%          dropout for regularization.
%       3. Output Layer: A fully connected layer followed by a tanh activation
%          to output actions bounded between -1 and 1.
%       The number of neurons in each hidden layer can be easily adjusted.
%       Dropout layers are included for regularization.
%
%   See also: rlDDPGAgent, rlRepresentation, createCriticNet

% Define layer parameters (input, output, hidden layers, dropout)
inputDimension = 1; % 1D input representing the state
outputDimension = 1; % 1D output representing the action
hiddenLayerSizes = [256, 128, 64, 32]; % Sizes of hidden layers
dropoutRate = 0.2; % Dropout rate to regularize the network

% Build the network
actorNet = buildActorNetwork(inputDimension, outputDimension, hiddenLayerSizes, dropoutRate);

end

% Local function to build the deep actor network
function lgraph = buildActorNetwork(inputDim, outputDim, hiddenSizes, dropoutRate)
%BUILDActorNetwork Constructs the actor network with specified layers
%
%   lgraph = BUILDActorNetwork(inputDim, outputDim, hiddenSizes, dropoutRate)
%   constructs a layer graph representing a deep actor network architecture.
%   This function defines the layers, connects them, and returns a layerGraph.
%
%   Inputs:
%       inputDim - Integer, the dimension of the input (state)
%       outputDim - Integer, the dimension of the output (action)
%       hiddenSizes - Array of integers, the number of neurons in each hidden layer
%       dropoutRate - Scalar, the rate of dropout regularization
%
%   Outputs:
%       lgraph - A LayerGraph object containing the actor network

% Initialize the layers array with the input layer
layers = [
    featureInputLayer(inputDim, 'Normalization', 'none', 'Name', 'state')
];

% Add hidden layers with ReLU activation and dropout
for i = 1:numel(hiddenSizes)
    layers = [
        layers
        fullyConnectedLayer(hiddenSizes(i), 'WeightsInitializer', 'he', 'Name', ['fc' num2str(i)])
        reluLayer('Name', ['relu' num2str(i)])
        dropoutLayer(dropoutRate, 'Name', ['dropout' num2str(i)])
    ];
end

% Add the output layer and tanh activation for action bounds
layers = [
    layers
    fullyConnectedLayer(outputDim, 'Name', 'fcOut')
    tanhLayer('Name', 'actorOutput')
    ];

% Assemble the layers into a layer graph
lgraph = layerGraph(layers);

end
