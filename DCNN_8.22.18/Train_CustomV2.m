%% Train_CustomV2.m
% Phase 2 (iteration v2): improved from-scratch CNN addressing v1's underfitting.
% Changes vs v1: light flip-only augmentation (no rotation/large shifts that mask
% splicing cues), deeper conv stack, an FC head (preserves spatial detail instead
% of global-average-pooling it away), higher LR, more epochs.
% Produces the full metric suite + train/val loss & accuracy curves.
%
% Run: matlab -batch "VARIANT='v2'; Train_CustomV2"

clc; close all;
if ~exist('VARIANT','var') || isempty(VARIANT); VARIANT = 'v2'; end
rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir); rootDir = pwd; end
cd(rootDir);

resDir = fullfile(rootDir, 'Results', ['Optimized_' VARIANT]);
if ~exist(resDir, 'dir'); mkdir(resDir); end
diary(fullfile(resDir, 'run_log.txt')); diary on;
fprintf('==================================================\n');
fprintf(' Phase 2 - Custom CNN  [%s]\n', VARIANT);
fprintf(' Date: %s\n', datestr(now));
fprintf('==================================================\n');

%% Data
dbPath = fullfile(rootDir, 'Database');
IMDS = imageDatastore(dbPath, 'IncludeSubfolders', true, ...
    'FileExtensions', '.jpg', 'LabelSource', 'foldernames');
fprintf('Total images found: %d\n', numel(IMDS.Files));
IMDS = filterReadable(IMDS);
classNames = string(categories(IMDS.Labels));
numClasses = numel(classNames);
fprintf('Total readable images: %d | Classes: %s\n', numel(IMDS.Files), strjoin(classNames, ', '));
disp(countEachLabel(IMDS));

rng(123);
[trainDS, valDS] = splitEachLabel(IMDS, 0.8, 'randomized');
inputSize = [114 114 3];

% Light augmentation only (flips + tiny shifts)
augmenter = imageDataAugmenter('RandXReflection', true, 'RandYReflection', true, ...
    'RandXTranslation', [-4 4], 'RandYTranslation', [-4 4]);
augTrain = augmentedImageDatastore(inputSize, trainDS, ...
    'DataAugmentation', augmenter, 'ColorPreprocessing', 'gray2rgb');
augVal   = augmentedImageDatastore(inputSize, valDS, 'ColorPreprocessing', 'gray2rgb');
augTrainEval = augmentedImageDatastore(inputSize, trainDS, 'ColorPreprocessing', 'gray2rgb');

%% Architecture: deeper conv stack + FC head
layers = [
    imageInputLayer(inputSize, 'Name','input', 'Normalization','zerocenter')

    convolution2dLayer(3, 32,  'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','bn1'); reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','pool1')

    convolution2dLayer(3, 64,  'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2'); reluLayer('Name','relu2')
    maxPooling2dLayer(2,'Stride',2,'Name','pool2')

    convolution2dLayer(3, 128, 'Padding','same','Name','conv3')
    batchNormalizationLayer('Name','bn3'); reluLayer('Name','relu3')
    maxPooling2dLayer(2,'Stride',2,'Name','pool3')

    convolution2dLayer(3, 256, 'Padding','same','Name','conv4')
    batchNormalizationLayer('Name','bn4'); reluLayer('Name','relu4')
    maxPooling2dLayer(2,'Stride',2,'Name','pool4')

    dropoutLayer(0.4,'Name','drop1')
    fullyConnectedLayer(256,'Name','fc1'); reluLayer('Name','relu_fc')
    dropoutLayer(0.4,'Name','drop2')
    fullyConnectedLayer(numClasses,'Name','fc2')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')];

%% Training options
valFreq = max(1, floor(numel(trainDS.Files)/64));
opts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'L2Regularization', 1e-4, ...
    'ValidationData', augVal, ...
    'ValidationFrequency', valFreq, ...
    'ValidationPatience', 8, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, 'LearnRateDropPeriod',15, ...
    'Verbose', true, 'VerboseFrequency', valFreq, ...
    'ExecutionEnvironment', 'auto', ...
    'OutputNetwork', 'best-validation-loss', ...
    'Plots', 'none');

fprintf('\nTraining...\n');
[net, info] = trainNetwork(augTrain, layers, opts);
save(fullfile(resDir, ['XONet_' VARIANT '.mat']), 'net', 'info');

%% Curves
f = figure('Visible','off','Position',[100 100 800 600]);
plot(info.TrainingLoss,'b-','LineWidth',1.2); hold on;
vi=find(~isnan(info.ValidationLoss)); plot(vi,info.ValidationLoss(vi),'r-o','LineWidth',1.5,'MarkerSize',4);
xlabel('Iteration'); ylabel('Loss'); grid on; title('Training vs Validation Loss');
legend({'Training Loss','Validation Loss'}); saveas(f,fullfile(resDir,'loss_curve.png')); close(f);
f = figure('Visible','off','Position',[100 100 800 600]);
plot(info.TrainingAccuracy,'b-','LineWidth',1.2); hold on;
ai=find(~isnan(info.ValidationAccuracy)); plot(ai,info.ValidationAccuracy(ai),'r-o','LineWidth',1.5,'MarkerSize',4);
xlabel('Iteration'); ylabel('Accuracy (%)'); grid on; title('Training vs Validation Accuracy');
legend({'Training Accuracy','Validation Accuracy'},'Location','southeast'); saveas(f,fullfile(resDir,'accuracy_curve.png')); close(f);

%% Evaluate
[predTrain, scoreTrain] = classify(net, augTrainEval, 'MiniBatchSize', 64);
[predVal, scoreVal]     = classify(net, augVal, 'MiniBatchSize', 64);
yTrain = trainDS.Labels; yVal = valDS.Labels;
mTrain = computeMetrics(yTrain, predTrain, classNames);
mVal   = computeMetrics(yVal,   predVal,   classNames);
printMetrics('TRAINING SET',   mTrain);
printMetrics('VALIDATION SET', mVal);
saveConfusion(yTrain, predTrain, ['Confusion - Training (' VARIANT ')'], fullfile(resDir,'confusion_training.png'));
saveConfusion(yVal,   predVal,   ['Confusion - Validation (' VARIANT ')'], fullfile(resDir,'confusion_validation.png'));
posCol = numClasses; posClass = classNames(posCol);
[AUCval, AUCtr] = plotROC(yVal, scoreVal, yTrain, scoreTrain, classNames, posCol, fullfile(resDir,'roc_curve.png'));
fprintf('\nAUC  - Validation: %.4f | Training: %.4f  (positive class = %s)\n', AUCval, AUCtr, posClass);

results = struct('train', mTrain, 'val', mVal, 'AUC_val', AUCval, 'AUC_train', AUCtr, ...
    'posClass', posClass, 'classNames', {classNames}, 'inputSize', inputSize, 'variant', VARIANT);
save(fullfile(resDir, 'results_optimized.mat'), 'results', 'info');
writeSummary(fullfile(resDir, 'SUMMARY.txt'), ['Optimized ' VARIANT], mTrain, mVal, AUCtr, AUCval, posClass);
fprintf('\nAll Phase-2 (%s) outputs saved to: %s\n', VARIANT, resDir);
fprintf('DONE_TRAIN_OPTIMIZED\n');
diary off;
