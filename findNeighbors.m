
function neighbors = findNeighbors(nodeID, nodes, communicationRange)
    % Calculate distances from the current node to all other nodes
    distances = sqrt(sum((nodes.position - nodes.position(nodeID, :)).^2, 2));
    % Find neighbors within the communication range, excluding the node itself and base station
    neighbors = find(distances <= communicationRange & distances > 0 & (1:100)');
end
