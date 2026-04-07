% Function to plot the 3D network dynamically
function plotNetwork3D(nodes, baseStation, iteration, sourceNode, sourcePath)
    figure(1); clf;

    % Plot all nodes 
    scatter3(nodes.position(:, 1), nodes.position(:, 2), nodes.position(:, 3), ...
             40, nodes.energy, 'm', 'filled');
    hold on;

    % Highlight the source node
    scatter3(nodes.position(sourceNode, 1), nodes.position(sourceNode, 2), nodes.position(sourceNode, 3), ...
             100, 'g', 'filled', 'MarkerEdgeColor', 'k');

    % Highlight the base station
    scatter3(baseStation(1), baseStation(2), baseStation(3), ...
             100, 'r', 'filled');

    % Plot the source node's path
    plot3(sourcePath(:, 1), sourcePath(:, 2), sourcePath(:, 3), 'b-', 'LineWidth', 2);

    % Add title and labels
    xlabel('X (meters)');
    ylabel('Y (meters)');
    zlabel('Z (meters)');
    grid on;
    view(45, 30);
    drawnow;
end
