function reward = computeReward(currentNode, selectedNeighbor, nodes, baseStation,speedOfSound, energyThreshold, depthInfo)
    % Parameters
    a1 = 0.1; % Weight for neighbor out of range
    a2 = 0.1; % Weight for invalid action
    a3 = 0.2; % Weight for selecting destination node
    a4 = 0.1; % Weight for normal forwarding
    depthWeight = 0.1; % Weight for depth information
    nmfWeight = 0.1;   % Weight for Node Mobility Factor (NMF)
    nrsWeight = 0.1;   % Weight for Node Reputation Score (NRS)
    ailWeight = 0.1;   % Weight for Acoustic Interference Level (AIL)
    energyWeight = 0.1; % Weight for residual energy

    % Reward Initialization
    reward = 0;

    % Compute distance to base station
    distanceToBaseStation = norm(nodes.position(selectedNeighbor, :) - baseStation);

    % Check if selected neighbor is within communication range
    if distanceToBaseStation <= 200 % Communication range
    % Positive reward for being within range
    reward = reward + a1; % Reward based on weight
    else
    % Penalize for being out of range
    reward = reward - a1;
    % return;
    end

    % Check for invalid action
    if nodes.energy(selectedNeighbor) <= energyThreshold
        % Penalize for invalid action
        reward = reward - a2;
    else
        % Positive reward 
        reward = reward + a2;
    % return;
    end

    % Check if destination node is reached
    if distanceToBaseStation < 0.1 % Stop threshold
        % positive for reaching the destination node
        reward = reward + a3;
    else
        % Penalize for not reaching the destination
        reward = reward - a3;
    % return;
    end

 % Normal forwarding reward (applies only for valid forwarding)
    if distanceToBaseStation > 0.1 && distanceToBaseStation <= 200 && nodes.energy(selectedNeighbor) > energyThreshold
        % Add normal forwarding reward
        reward = reward + a4;
    else
        % Penalize invalid forwarding
        reward = reward - a4;
    end
  end
