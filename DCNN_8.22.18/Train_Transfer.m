%% Train_Transfer.m
% Phase 2 (alternative): Transfer learning from a pretrained CNN backbone.
% Produces the full metric suite + train/val loss & accuracy curves.
%
% Run, e.g.:  matlab -batch "BASENET='resnet18'; Train_Transfer"
% Supported BASENET: 'resnet18','googlenet','squeezenet','alexnet' (whichever
% support packages are installed; the script reports if one is missing).

clc; close all;
if ~exist('BASENET','var') || isempty(BASENET); BASENET = 'resnet18'; end
if ~exist('EPOCHS','var')  || isempty(EPOCHS);  EPOCHS  = 12;   end
if ~exist('LR0','var')     || isempty(LR0);     LR0     = 1e-4; end
if ~exist('MBS','var')     || isempty(MBS);     MBS     = 64;   end
if ~exist('TAG','var')     || isempty(TAG);     TAG     = BASENET; end
rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir); rootDir = pwd; end
cd(rootDir);

resDir = fullfile(rootDir, 'Results', ['Transfer_' TAG]);
if ~exist(resDir, 'dir'); mkdir(resDir); end
diary(fullfile(resDir, 'run_log.txt')); diary on;
fprintf('==================================================\n');
fprintf(' Phase 2 - Transfer learning  [%s]\n', BASENET);
fprintf(' Date: %s\n', datestr(now));
fprintf('==================================================\n');

%% Load pretrained backbone (graceful failure if package missing)
setupPretrained();   % add locally-cached support packages (resnet18/50, alexnet, vgg16) to path
try
    switch lower(BASENET)
        case 'resnet18';   baseNet = resnet18;
        case 'resnet50';   baseNet = resnet50;
        case 'googlenet';  baseNet = googlenet;
        case 'squeezenet'; baseNet = squeezenet;
        case 'alexnet';    baseNet = alexnet;
        case 'vgg16';      baseNet = vgg16;
        otherwise; error('Unknown BASENET %s', BASENET);
    end
catch ME
    fprintf(2, 'Could not load %s: %s\n', BASENET, ME.message);
    fprintf(2, 'Install via Add-On Explorer: "Deep Learning Toolbox Model for %s".\n', BASENET);
    diary off; error('Missing pretrained network support package for %s.', BASENET);
end

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

rng(123);  % SAME split seed as Phase-1 for a fair comparison
[trainDS, valDS] = splitEachLabel(IMDS, 0.8, 'randomized');

inputSize = baseNet.Layers(1).InputSize;

% Light augmentation only: flips + small shifts. Rotation/scaling introduce
% interpolation artifacts that can mask the very splicing cues we must detect.
augmenter = imageDataAugmenter('RandXReflection', true, 'RandYReflection', true, ...
    'RandXTranslation', [-4 4], 'RandYTranslation', [-4 4]);
augTrain = augmentedImageDatastore(inputSize, trainDS, ...
    'DataAugmentation', augmenter, 'ColorPreprocessing', 'gray2rgb');
augVal   = augmentedImageDatastore(inputSize, valDS, 'ColorPreprocessing', 'gray2rgb');
augTrainEval = augmentedImageDatastore(inputSize, trainDS, 'ColorPreprocessing', 'gray2rgb');

%% Replace classification head for 2 classes
lgraph = layerGraph(baseNet);
[learnableLayer, classLayer] = findHeadLayers(lgraph);
newLearnable = fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
if isa(learnableLayer, 'nnet.cnn.layer.Convolution2DLayer')
    newLearnable = convolution2dLayer(1, numClasses, 'Name', 'new_conv', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
end
lgraph = replaceLayer(lgraph, learnableLayer.Name, newLearnable);
lgraph = replaceLayer(lgraph, classLayer.Name, classificationLayer('Name','new_output'));

%% Training options (capture history)
valFreq = max(1, floor(numel(trainDS.Files)/MBS));
opts = trainingOptions('adam', ...
    'InitialLearnRate', LR0, ...
    'MaxEpochs', EPOCHS, ...
    'MiniBatchSize', MBS, ...
    'Shuffle', 'every-epoch', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 12, ...
    'ValidationData', augVal, ...
    'ValidationFrequency', valFreq, ...
    'ValidationPatience', 8, ...
    'Verbose', true, 'VerboseFrequency', valFreq, ...
    'ExecutionEnvironment', 'auto', ...
    'OutputNetwork', 'best-validation-loss', ...
    'Plots', 'none');

fprintf('\nTraining (transfer)...\n');
[net, info] = trainNetwork(augTrain, lgraph, opts);
save(fullfile(resDir, ['net_' BASENET '.mat']), 'net', 'info', '-v7.3');

%% Curves
plotTrainCurves(info, resDir);

%% Evaluate
[predTrain, scoreTrain] = classify(net, augTrainEval, 'MiniBatchSize', MBS);
[predVal, scoreVal]     = classify(net, augVal, 'MiniBatchSize', MBS);
yTrain = trainDS.Labels; yVal = valDS.Labels;

mTrain = computeMetrics(yTrain, predTrain, classNames);
mVal   = computeMetrics(yVal,   predVal,   classNames);
printMetrics('TRAINING SET',   mTrain);
printMetrics('VALIDATION SET', mVal);

saveConfusion(yTrain, predTrain, ['Confusion - Training (' BASENET ')'], fullfile(resDir,'confusion_training.png'));
saveConfusion(yVal,   predVal,   ['Confusion - Validation (' BASENET ')'], fullfile(resDir,'confusion_validation.png'));

posCol = numClasses; posClass = classNames(posCol);
[AUCval, AUCtr] = plotROC(yVal, scoreVal, yTrain, scoreTrain, classNames, posCol, fullfile(resDir,'roc_curve.png'));
fprintf('\nAUC  - Validation: %.4f | Training: %.4f  (positive class = %s)\n', AUCval, AUCtr, posClass);

results = struct('train', mTrain, 'val', mVal, 'AUC_val', AUCval, 'AUC_train', AUCtr, ...
    'posClass', posClass, 'classNames', {classNames}, 'inputSize', inputSize, 'basenet', BASENET);
save(fullfile(resDir, 'results_transfer.mat'), 'results', 'info');
writeSummary(fullfile(resDir, 'SUMMARY.txt'), ['Transfer ' BASENET], mTrain, mVal, AUCtr, AUCval, posClass);

fprintf('\nAll transfer (%s) outputs saved to: %s\n', BASENET, resDir);
fprintf('DONE_TRAIN_OPTIMIZED\n');
diary off;

%% ---- helpers ----
function [learnableLayer, classLayer] = findHeadLayers(lgraph)
    layers = lgraph.Layers;
    classLayer = layers(end);
    learnableLayer = [];
    for i = numel(layers):-1:1
        if isa(layers(i),'nnet.cnn.layer.FullyConnectedLayer') || ...
           isa(layers(i),'nnet.cnn.layer.Convolution2DLayer')
            learnableLayer = layers(i); break;
        end
    end
end

function plotTrainCurves(info, resDir)
    f = figure('Visible','off','Position',[100 100 800 600]);
    plot(info.TrainingLoss,'b-','LineWidth',1.2); hold on;
    vi = find(~isnan(info.ValidationLoss));
    plot(vi, info.ValidationLoss(vi),'r-o','LineWidth',1.5,'MarkerSize',4);
    xlabel('Iteration'); ylabel('Loss'); grid on; title('Training vs Validation Loss');
    legend({'Training Loss','Validation Loss'}); saveas(f,fullfile(resDir,'loss_curve.png')); close(f);
    f = figure('Visible','off','Position',[100 100 800 600]);
    plot(info.TrainingAccuracy,'b-','LineWidth',1.2); hold on;
    ai = find(~isnan(info.ValidationAccuracy));
    plot(ai, info.ValidationAccuracy(ai),'r-o','LineWidth',1.5,'MarkerSize',4);
    xlabel('Iteration'); ylabel('Accuracy (%)'); grid on; title('Training vs Validation Accuracy');
    legend({'Training Accuracy','Validation Accuracy'},'Location','southeast');
    saveas(f,fullfile(resDir,'accuracy_curve.png')); close(f);
end
