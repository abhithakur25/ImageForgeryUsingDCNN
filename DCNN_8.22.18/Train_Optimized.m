%% Train_Optimized.m
% Phase 2: Train an IMPROVED model and produce the full metric suite, including
% the train/validation LOSS and ACCURACY curves (captured from trainNetwork info).
%
% Improvements over the original DCNN:
%   * Data augmentation (random reflection + translation) to reduce overfitting
%   * Modern conv blocks with Batch Normalization for stable, faster convergence
%   * Dropout before the classifier for regularization
%   * Adam optimizer with validation-based monitoring + early stopping
%   * Reproducible split (same seed as the Phase-1 test for fair comparison)
%
% Run a specific variant from batch mode, e.g.:
%   matlab -batch "VARIANT='v1'; Train_Optimized"
% Defaults to 'v1' if VARIANT is undefined.

clc; close all;
if ~exist('VARIANT','var') || isempty(VARIANT); VARIANT = 'v1'; end
rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir); rootDir = pwd; end
cd(rootDir);

resDir = fullfile(rootDir, 'Results', ['Optimized_' VARIANT]);
if ~exist(resDir, 'dir'); mkdir(resDir); end
diary(fullfile(resDir, 'run_log.txt')); diary on;
fprintf('==================================================\n');
fprintf(' Phase 2 - Training optimized model  [%s]\n', VARIANT);
fprintf(' Date: %s\n', datestr(now));
fprintf('==================================================\n');

%% 1. Data
dbPath = fullfile(rootDir, 'Database');
IMDS = imageDatastore(dbPath, 'IncludeSubfolders', true, ...
    'FileExtensions', '.jpg', 'LabelSource', 'foldernames');
fprintf('Total images found: %d\n', numel(IMDS.Files));
IMDS = filterReadable(IMDS);   % drop corrupt/unreadable images
classNames = string(categories(IMDS.Labels));
numClasses = numel(classNames);
fprintf('Total readable images: %d | Classes: %s\n', numel(IMDS.Files), strjoin(classNames, ', '));
disp(countEachLabel(IMDS));

rng(123);  % SAME split seed as Phase-1 for a fair comparison
[trainDS, valDS] = splitEachLabel(IMDS, 0.8, 'randomized');

inputSize = [114 114 3];

% Augmentation on training data only; plain resize for validation
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandRotation', [-10 10], ...
    'RandXTranslation', [-6 6], ...
    'RandYTranslation', [-6 6]);
augTrain = augmentedImageDatastore(inputSize, trainDS, ...
    'DataAugmentation', augmenter, 'ColorPreprocessing', 'gray2rgb');
augVal   = augmentedImageDatastore(inputSize, valDS, 'ColorPreprocessing', 'gray2rgb');
% Non-augmented training view for final metric reporting
augTrainEval = augmentedImageDatastore(inputSize, trainDS, 'ColorPreprocessing', 'gray2rgb');

%% 2. Network architecture (improved)
layers = [
    imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'zerocenter')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name','bn1'); reluLayer('Name','relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name','pool1')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name','bn2'); reluLayer('Name','relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name','pool2')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name','bn3'); reluLayer('Name','relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name','pool3')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name','bn4'); reluLayer('Name','relu4')

    globalAveragePooling2dLayer('Name','gap')
    dropoutLayer(0.5, 'Name','drop')
    fullyConnectedLayer(numClasses, 'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')];

%% 3. Training options (capture validation history -> loss/accuracy curves)
valFreq = max(1, floor(numel(trainDS.Files)/64));
opts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'L2Regularization', 1e-4, ...
    'ValidationData', augVal, ...
    'ValidationFrequency', valFreq, ...
    'ValidationPatience', 6, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 12, ...
    'Verbose', true, 'VerboseFrequency', valFreq, ...
    'ExecutionEnvironment', 'auto', ...
    'OutputNetwork', 'best-validation-loss', ...
    'Plots', 'none');

%% 4. Train
fprintf('\nTraining...\n');
[net, info] = trainNetwork(augTrain, layers, opts);
save(fullfile(resDir, ['XONet_' VARIANT '.mat']), 'net', 'info');

%% 5. Training/validation LOSS and ACCURACY curves
plotCurves(info, resDir);

%% 6. Evaluate (capture scores for ROC/AUC)
fprintf('\nClassifying training set...\n');
[predTrain, scoreTrain] = classify(net, augTrainEval, 'MiniBatchSize', 64);
fprintf('Classifying validation set...\n');
[predVal, scoreVal]     = classify(net, augVal, 'MiniBatchSize', 64);
yTrain = trainDS.Labels; yVal = valDS.Labels;

mTrain = computeMetrics(yTrain, predTrain, classNames);
mVal   = computeMetrics(yVal,   predVal,   classNames);
printMetrics('TRAINING SET',   mTrain);
printMetrics('VALIDATION SET', mVal);

saveConfusion(yTrain, predTrain, ['Confusion Matrix - Training (' VARIANT ')'], ...
    fullfile(resDir, 'confusion_training.png'));
saveConfusion(yVal, predVal, ['Confusion Matrix - Validation (' VARIANT ')'], ...
    fullfile(resDir, 'confusion_validation.png'));

posCol = numClasses; posClass = classNames(posCol);
[AUCval, AUCtr] = plotROC(yVal, scoreVal, yTrain, scoreTrain, classNames, posCol, ...
    fullfile(resDir, 'roc_curve.png'));
fprintf('\nAUC  - Validation: %.4f | Training: %.4f  (positive class = %s)\n', ...
    AUCval, AUCtr, posClass);

results = struct('train', mTrain, 'val', mVal, 'AUC_val', AUCval, 'AUC_train', AUCtr, ...
    'posClass', posClass, 'classNames', {classNames}, 'inputSize', inputSize, ...
    'variant', VARIANT, 'finalValLoss', info.FinalValidationLoss);
save(fullfile(resDir, 'results_optimized.mat'), 'results', 'info');
writeSummary(fullfile(resDir, 'SUMMARY.txt'), ['Optimized ' VARIANT], ...
    mTrain, mVal, AUCtr, AUCval, posClass);

fprintf('\nAll Phase-2 (%s) outputs saved to: %s\n', VARIANT, resDir);
fprintf('DONE_TRAIN_OPTIMIZED\n');
diary off;

%% ---- local helper for curves ----
function plotCurves(info, resDir)
    % Loss curve
    f = figure('Visible','off','Position',[100 100 800 600]);
    plot(info.TrainingLoss, 'b-', 'LineWidth', 1.2); hold on;
    vidx = find(~isnan(info.ValidationLoss));
    plot(vidx, info.ValidationLoss(vidx), 'r-o', 'LineWidth', 1.5, 'MarkerSize', 4);
    xlabel('Iteration'); ylabel('Loss'); grid on;
    title('Training vs Validation Loss');
    legend({'Training Loss','Validation Loss'}, 'Location','northeast');
    saveas(f, fullfile(resDir, 'loss_curve.png')); close(f);

    % Accuracy curve
    f = figure('Visible','off','Position',[100 100 800 600]);
    plot(info.TrainingAccuracy, 'b-', 'LineWidth', 1.2); hold on;
    aidx = find(~isnan(info.ValidationAccuracy));
    plot(aidx, info.ValidationAccuracy(aidx), 'r-o', 'LineWidth', 1.5, 'MarkerSize', 4);
    xlabel('Iteration'); ylabel('Accuracy (%)'); grid on;
    title('Training vs Validation Accuracy');
    legend({'Training Accuracy','Validation Accuracy'}, 'Location','southeast');
    saveas(f, fullfile(resDir, 'accuracy_curve.png')); close(f);
end
