%% Test_Existing_Model.m
% Phase 1: Evaluate the EXISTING trained model (XONet_r003.mat) WITHOUT retraining.
% Produces: confusion matrix, precision, recall, F1 (per-class, macro, weighted),
% training-set accuracy, validation-set accuracy, ROC curve, AUC.
%
% NOTE on loss curves: training/validation LOSS and ACCURACY *curves* require the
% training-info struct returned as the 2nd output of trainNetwork(). The original
% code discarded it (only the network was saved), so loss curves cannot be
% reconstructed from a saved network. They are produced in Phase 2 (retraining,
% see Train_Optimized.m). This script reports everything obtainable from the
% saved model.

clc; close all;
rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir); rootDir = pwd; end
cd(rootDir);

resDir = fullfile(rootDir, 'Results', 'Existing_Model');
if ~exist(resDir, 'dir'); mkdir(resDir); end

diary(fullfile(resDir, 'run_log.txt')); diary on;
fprintf('==================================================\n');
fprintf(' Phase 1 - Testing existing model XONet_r003.mat\n');
fprintf(' Date: %s\n', datestr(now));
fprintf('==================================================\n');

%% 1. Load existing trained network
S = load('XONet_r003.mat');
assert(isfield(S,'XONet'), 'XONet variable not found in XONet_r003.mat');
net = S.XONet;
inputSize = net.Layers(1).InputSize;
classNames = string(net.Layers(end).Classes(:));
fprintf('Loaded network (%s). Input size: %s. Classes: %s\n', ...
    class(net), mat2str(inputSize), strjoin(classNames, ', '));

%% 2. Build datastore (correct local path) and reproducible split
dbPath = fullfile(rootDir, 'Database');
IMDS = imageDatastore(dbPath, 'IncludeSubfolders', true, ...
    'FileExtensions', '.jpg', 'LabelSource', 'foldernames');
fprintf('Total images found: %d\n', numel(IMDS.Files));
IMDS = filterReadable(IMDS);   % drop corrupt/unreadable images
fprintf('Total readable images: %d\n', numel(IMDS.Files));
disp(countEachLabel(IMDS));

rng(123);   % fixed seed so the split is reproducible
[trainDS, valDS] = splitEachLabel(IMDS, 0.8, 'randomized');
fprintf('Train images: %d | Validation images: %d\n', ...
    numel(trainDS.Files), numel(valDS.Files));

% Resize-on-read wrappers so images always match the network input size
cp = 'gray2rgb';
if numel(inputSize) == 3 && inputSize(3) == 1; cp = 'rgb2gray'; end
augTrain = augmentedImageDatastore(inputSize, trainDS, 'ColorPreprocessing', cp);
augVal   = augmentedImageDatastore(inputSize, valDS,   'ColorPreprocessing', cp);

%% 3. Classify (capture scores for ROC/AUC)
fprintf('\nClassifying training set...\n');
[predTrain, scoreTrain] = classify(net, augTrain, 'MiniBatchSize', 64);
fprintf('Classifying validation set...\n');
[predVal, scoreVal]     = classify(net, augVal,   'MiniBatchSize', 64);

yTrain = trainDS.Labels;
yVal   = valDS.Labels;

%% 4. Metrics
mTrain = computeMetrics(yTrain, predTrain, classNames);
mVal   = computeMetrics(yVal,   predVal,   classNames);
printMetrics('TRAINING SET',   mTrain);
printMetrics('VALIDATION SET', mVal);

%% 5. Confusion matrix figures
saveConfusion(yTrain, predTrain, 'Confusion Matrix - Training Set', ...
    fullfile(resDir, 'confusion_training.png'));
saveConfusion(yVal, predVal, 'Confusion Matrix - Validation Set', ...
    fullfile(resDir, 'confusion_validation.png'));

%% 6. ROC + AUC (positive class = last class, e.g. 'Sp'/forgery)
posCol   = numel(classNames);
posClass = classNames(posCol);
[AUCval, AUCtr] = plotROC(yVal, scoreVal, yTrain, scoreTrain, classNames, posCol, ...
    fullfile(resDir, 'roc_curve.png'));
fprintf('\nAUC  - Validation: %.4f | Training: %.4f  (positive class = %s)\n', ...
    AUCval, AUCtr, posClass);

%% 7. Save summary report + .mat
results = struct('train', mTrain, 'val', mVal, ...
    'AUC_val', AUCval, 'AUC_train', AUCtr, 'posClass', posClass, ...
    'classNames', {classNames}, 'inputSize', inputSize, 'model', 'XONet_r003.mat');
save(fullfile(resDir, 'results_existing_model.mat'), 'results');
writeSummary(fullfile(resDir, 'SUMMARY.txt'), 'XONet_r003.mat (existing model)', ...
    mTrain, mVal, AUCtr, AUCval, posClass);

fprintf('\nAll Phase-1 outputs saved to: %s\n', resDir);
fprintf('DONE_TEST_EXISTING\n');
diary off;
