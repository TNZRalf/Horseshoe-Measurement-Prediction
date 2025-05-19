% Load and preprocess data
p = readtable('Horseshoe.csv.csv','ReadVariableNames',false);
p = rmmissing(p);  % Remove rows with missing values
data_mean = mean(p);
data_std = std(p);
n_data = (p - data_mean) ./ data_std;

% Separate inputs and outputs
input = n_data{:,1:4}';
output = n_data{:,5}';
trainingfunction = 'trainbr';

% Parameter configurations for tuning
learningRates = [0.1];  % Learning rates to test
epochs = [50];  % Number of epochs to test
tolerance = 0.5;  % Accuracy tolerance for prediction
numRuns = 10;  % Number of times to repeat training/testing

% Initialize variables for tracking
allResults = [];  % Store performance results for each configuration and run
tic;  % Start timing for entire training process

% Main loop to run the entire process 10 times
for run = 1:numRuns
    fprintf('Run %d/%d\n', run, numRuns);  % Track the current run

    iteration = 0;
    for lr = learningRates  % Loop over learning rates
        for epoch = epochs  % Loop over epochs
            for hiddenUnitsLayer1 = 5  % Hidden layer 1 units
                for hiddenUnitsLayer2 = 10  % Hidden layer 2 units
                    % Define and configure network
                    net = fitnet([hiddenUnitsLayer1, hiddenUnitsLayer2], trainingfunction);
                    net.trainParam.lr = lr;
                    net.trainParam.epochs = epoch;
                    net.divideFcn = 'dividerand';
                    net.performFcn = 'mse';
                    net.divideParam.trainRatio = 0.7;
                    net.divideParam.valRatio = 0.15;
                    net.divideParam.testRatio = 0.15;
                    net.trainParam.showWindow = false;  % Disable training GUI window

                    % Split data and train
                    [trainInd, valInd, testInd] = dividerand(size(input, 2), 0.7, 0.15, 0.15);
                    trainInput = input(:, trainInd);
                    trainOutput = output(:, trainInd);
                    testInput = input(:, testInd);
                    testOutput = output(:, testInd);

                    % Train and evaluate network
                    net = train(net, trainInput, trainOutput, 'UseGpu', 'no');
                    y = net(testInput);
                    mse = perform(net, y, testOutput);
                    accuracy = sum(abs(y - testOutput) <= tolerance) / numel(testOutput) * 100;

                    % Store results for each configuration and run
                    resultStruct = struct('Run', run, 'LearningRate', lr, 'Epochs', epoch, ...
                                          'Layer1', hiddenUnitsLayer1, 'Layer2', hiddenUnitsLayer2, ...
                                          'MSE', mse, 'Accuracy', accuracy);
                    allResults = [allResults; resultStruct];

                    % Track iterations and print progress
                    iteration = iteration + 1;
                    fprintf('Iteration %d, Run %d: LR=%.3f, Epochs=%d, Layers=[%d, %d], MSE=%.4f, Accuracy=%.2f%%\n', ...
                            iteration, run, lr, epoch, hiddenUnitsLayer1, hiddenUnitsLayer2, mse, accuracy);
                end
            end
        end
    end
end

% Convert all results to a table for better display and calculate averages
allResultsTable = struct2table(allResults);

% Calculate average MSE and accuracy across all runs for each configuration
avgResults = varfun(@mean, allResultsTable, 'InputVariables', {'MSE', 'Accuracy'}, ...
                    'GroupingVariables', {'LearningRate', 'Epochs', 'Layer1', 'Layer2'});

% Display summary of results and timing
toc;
disp('Average Results Across 10 Runs:');
disp(avgResults);
