clc;
clear;
close all;
rng(1)

%% [NOTE : Original code]
% D2PG - Deep Deterministic Policy Gradient method 

alphaValues = [0.01 0.005 0.001];  
numNodeValues = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]; 

for a = 1:length(alphaValues) 
alpha = alphaValues(a);  

for i = 1:length(numNodeValues) 
numNodes = numNodeValues(i); 

fprintf('Iteration %d: NumNodes = %d, Alpha = %.4f\n', i, numNodes,alpha);

%% Simulation Parameters
networkDim = [500, 500, 500];
baseStation = [500, 500, 500];
initialEnergy = 100;
communicationRange = 200;
stopThreshold = 0.1;
speedOfSound = 1500;

v_min = 1;
v_max = 3;

channelCapacity = 10 * 1e3;
transmissionEnergy = 2;

PacketSize = randi([50, 100]);

receptionCoefficient = 0.1;
sourceNode = 5;

maxIterations = 100;

%% Q-learning Parameters
alphaQ = alpha;
gamma = 0.9;

SNR_threshold = 20;

alphaWeight = 0.5;
betaWeight = 0.3;
gammaWeight = 0.2;

%% Initialization

nodes.position = rand(numNodes, 3) .* networkDim;
nodes.energy = initialEnergy * ones(numNodes, 1);

nodes.position(numNodes, :) = baseStation;

depthInfo = nodes.position(:, 3);

nodes.mobility = v_min + (v_max - v_min) * rand(numNodes, 1);
nodes.ail = rand(numNodes, 1);
nodes.reputation = zeros(numNodes, 1);

%% AIL calculation
SNR = 10 + rand(numNodes, 1) * 30;
nodes.ail = SNR ./ SNR_threshold;

%% Node Reputation Score
PR = exp(-0.1 * (1:numNodes)');
CSR = 1 - exp(-0.05 * (1:numNodes)');
AR = exp(-0.2 * (1:numNodes)');

nodes.reputation = alphaWeight * PR + betaWeight * CSR + gammaWeight * (1 - AR);

%% Actor-Critic Initialization

actorNet = rand(numNodes, 1);
criticNet = zeros(numNodes, 1);

tau = 0.005;

targetActorNet = actorNet;
targetCriticNet = criticNet;

energyThreshold = mean(nodes.energy) * 0.5;

%% State Action Initialization

currentNode = sourceNode;

numStates = size(nodes.position, 1);
numActions = numStates;

Q = zeros(numStates, numActions);

%% Performance Metrics Initialization

sourcePath = nodes.position(currentNode, :);
visitedNodes = [];

packetsSent = PacketSize;
packetsReceived = 0;

totalDelay = 0;
transmissionCount = 0;

iteration = 0;

TotalcontrolPackets = 0;
controlPackets = 0;
dataPackets = 0;

%% ROUTING LOOP
tic;
while iteration < maxIterations

iteration = iteration + 1;

packetsSent = packetsSent + 1;
dataPackets = dataPackets + 1;

%% Neighbor Discovery

distances = vecnorm(nodes.position - nodes.position(currentNode, :), 2, 2);

neighbors = find(distances <= communicationRange & nodes.energy > 0);

neighbors = neighbors(neighbors ~= currentNode);

if isempty(neighbors)
break;
end

%% Neighbor Selection

baseStationDistances = vecnorm(nodes.position(neighbors, :) - baseStation, 2, 2);

[~, idx] = min(baseStationDistances);

selectedNeighbor = neighbors(idx);

if selectedNeighbor > numNodes
break;
end

%% Data Rate Check

dataRate = rand * channelCapacity;

if PacketSize * 8 > dataRate
break;
end

%% Network Load

loadRatio = sum(PacketSize * 8) / (channelCapacity * length(neighbors));

if loadRatio > 1
fprintf('Network overload: Load ratio exceeds capacity (%.2f).\n', loadRatio);
end

%% Reinforcement Learning Update

currentState = currentNode;
nextState = selectedNeighbor;

action = selectedNeighbor;

reward = computeReward(currentNode, selectedNeighbor, nodes, baseStation, speedOfSound, energyThreshold,depthInfo);

Q = updateQValue(Q, currentState, action, nextState, nodes, baseStation, speedOfSound, energyThreshold, alphaQ, gamma,depthInfo);

targetActorNet = softUpdateTargetNetwork(actorNet, targetActorNet, tau);
targetCriticNet = softUpdateTargetNetwork(criticNet, targetCriticNet, tau);

targetQValue = targetCriticNet(nextState);

tdError = reward + gamma * targetQValue - Q(currentState, action);

criticNet(currentNode) = criticNet(currentNode) + alphaQ * tdError;

%% Policy Gradient Update

actorPolicy = ones(numActions, maxIterations);
actorPolicy = actorPolicy ./ sum(actorPolicy, 2);

for a2 = 1:maxIterations

if a2 == action
grad = 1 - actorPolicy(currentState, a2);
else
grad = -actorPolicy(currentState, a2);
end

actorPolicy(currentState, a2) = actorPolicy(currentState, a2) + alpha * tdError * grad;

end

actorPolicy(currentState, :) = actorPolicy(currentState, :) / sum(actorPolicy(currentState, :)) + alphaQ * tdError;

[~, actorNet(currentNode)] = max(actorPolicy(currentState, :));

%% Delay Calculation

propagationDelay = norm(nodes.position(currentNode, :) - nodes.position(selectedNeighbor, :)) / speedOfSound;

totalDelay = totalDelay + propagationDelay;

transmissionCount = transmissionCount + 1;

%% Energy Consumption

transmissionEnergy = transmissionEnergy * (PacketSize / 100);

receptionEnergy = receptionCoefficient * PacketSize;

nodes.energy(currentNode) = nodes.energy(currentNode) - transmissionEnergy;

nodes.energy(selectedNeighbor) = nodes.energy(selectedNeighbor) - receptionEnergy;

%% Move to Next Node

visitedNodes = [visitedNodes currentNode];

sourcePath = [sourcePath; nodes.position(selectedNeighbor, :)];

currentNode = selectedNeighbor;

%% Buffer Update

Buffer = updateBuffer(iteration, currentState, action, reward, nextState);

nodes.energy(currentNode) = nodes.energy(currentNode) - transmissionEnergy;

%% Visualization

plotNetwork3D(nodes, baseStation, iteration, sourceNode, sourcePath);

%% Packet Counters

packetsReceived = PacketSize;

controlPackets = packetsReceived + 1;

TotalcontrolPackets = dataPackets + 1;

%% Destination Check

if norm(nodes.position(currentNode, :) - baseStation) < stopThreshold
break;
end

end

%% Performance Metrics

PDR = (packetsReceived / packetsSent);

totalEnergyConsumed = sum(initialEnergy * ones(numNodes, 1) - nodes.energy(1:numNodes));

D_avg = totalDelay / transmissionCount;

E_total = sum(nodes.energy(1:numNodes));

P_avg = mean(transmissionEnergy ./ nodes.energy(1:numNodes)) * 1000;

T_network = E_total / P_avg;

SuccessRate = 1 - (iteration / maxIterations);

%% Network Overhead

totalPackets = controlPackets + dataPackets;

networkOverhead = (TotalcontrolPackets / totalPackets) * 100;

%% Routing Overhead 

if packetsReceived == 0
RoutingOverhead = 0;
else
RoutingOverhead = TotalcontrolPackets / packetsReceived;
end

%% Computational Overhead (Processing Time / Total Packets)

TotalProcessingTime = toc;   % Total algorithm execution time

TotalPackets = dataPackets + controlPackets;

if TotalPackets == 0
    ComputationalOverhead = 0;
else
    ComputationalOverhead = TotalProcessingTime / TotalPackets;
end

%% Save Results

saveFilename = sprintf('result_iterations_%d_NumNodes_%d.mat', i, numNodes);

save(saveFilename, 'numNodeValues', 'alphaValues', 'numNodes', 'alpha', ...
'PDR','totalEnergyConsumed','D_avg','T_network','networkOverhead','RoutingOverhead','ComputationalOverhead');

saveFilename = sprintf('result_alpha_%.4f_NumNodes_%d.mat', alpha, numNodes);

save(saveFilename, 'numNodeValues', 'alphaValues', 'numNodes', 'alpha', ...
'PDR','totalEnergyConsumed','D_avg','T_network','networkOverhead','RoutingOverhead','ComputationalOverhead');

fprintf('Results saved to %s\n', saveFilename);

end
end

%% Plot Graphs

Result(numNodeValues, alphaValues);

%% FIN