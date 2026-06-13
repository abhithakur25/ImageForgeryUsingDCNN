doTraining = true;
close all;
load('training_001.mat')
net = training;
% layers = net.Layers

% layers(11)= fullyConnectedLayer(2);
% layers(13)= classificationLayer

IMDS = imageDatastore('F:\College\Abhishek\DCNN_NEW\Database\','IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames'); % use imageDatastore for loading the two image categories

example_image = readimage(IMDS,1);                      % read one example image
numChannels = size(example_image,3);                    % get color information
numImageCategories = size(categories(IMDS.Labels),1);   % get category labels
[trainingDS,validationDS] = splitEachLabel(IMDS,0.7,'randomize'); % generate training and validation set
LabelCnt = countEachLabel(IMDS);                        % load lable information
for cats=1:numImageCategories                           % print out how many images we have for each category
    fprintf('%s\t%d\n',LabelCnt.Label(cats),LabelCnt.Count(cats));
end

if doTraining
%% Setup of the CNN
    % Convolutional layer parameters
    filterSize = [5 5]
    numFilters = 32
    
    inputLayer = imageInputLayer(size(example_image));  % input layer with no data augmentation
    
    middleLayers = [
        convolution2dLayer(filterSize, numFilters, 'Padding', 4)
        reluLayer()
        maxPooling2dLayer(5, 'Stride', 2)
        convolution2dLayer(filterSize, numFilters*2, 'Padding', 4)
        reluLayer()
        maxPooling2dLayer(5, 'Stride',2)
        convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2)
        reluLayer()
        maxPooling2dLayer(3, 'Stride',2)];
        
    
    finalLayers = [
                    fullyConnectedLayer(numImageCategories)
                    softmaxLayer
                    classificationLayer
        
                   ];
    
    layers = [
        inputLayer
        middleLayers
        finalLayers
        ];
    
    %Initialize the first convolutional layer weights using normally distributed random numbers with standard deviation of 0.0001. This helps improve the convergence of training.
    layers(2).Weights = 0.0001 * randn([filterSize numChannels numFilters]);
    
    layers = layers(1:end-3);% Set the network training options end-3);

layers(end+1) = fullyConnectedLayer(numImageCategories);
layers(end+1) = reluLayer;
layers(end+1) = fullyConnectedLayer(numImageCategories);
layers(end+1) = softmaxLayer;
layers(end+1) = classificationLayer()


layers(end-2).WeightLearnRateFactor = 10;
layers(end-2).WeightL2Factor = 1;
layers(end-2).BiasLearnRateFactor = 20;
layers(end-2).BiasL2Factor = 0;


opts = trainingOptions('sgdm', ...
        'Momentum', 0.9, ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 10, ...
        'L2Regularization', 0.004, ...
        'MaxEpochs', 500, ...         % 10 for Quadro 
        'MiniBatchSize', 100, ...    % 64 for Quadro 
        'Verbose', true,'ExecutionEnvironment','multi-gpu','Plots','training-progress');

    
    
    % Train a network.
%%
        rng('default');
    rng(123); % random seed
    XONet = trainNetwork(IMDS, layers, opts);
    save('XONet_r002.mat','XONet');
   
else
        % Load pre-trained detector for the example.
        load('XONet_r002.mat');
   end

%% test the performance
% test network performance on validation set
[labels,~] = classify(XONet, validationDS, 'MiniBatchSize', 20);
confMat = confusionmat(validationDS.Labels, labels);
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
fprintf('Performance on validation set \t\t\t%.4f\n',mean(diag(confMat)));


if show.wrong_classified
    % % Find wrong classified examples
    idx = find(validationDS.Labels~=labels);
    idx2 = int16(rand(length(idx),1)*size(labels,1));
    fprintf('We have %d/%d wrong classifications\n',length(idx),size(labels,1));
        for jj = 1:1
        %%
        img = readimage(validationDS,idx(jj));
        imwrite(img, 'im1.jpg');
        folder = 'F:\College\Abhishek\DCNN_NEW';
        baseFileName = 'im1.jpg';
        deconvolution(folder,baseFileName, idx,jj, labels);
        end   
       
            %%
        for jj = 50:52
        img = readimage(validationDS,idx2(jj));
        imwrite(img, 'im2.jpg');
        folder = 'F:\College\Abhishek\DCNN_NEW';
        baseFileName = 'im2.jpg';
        deconvolution(folder,baseFileName, idx2,jj, labels);
        
        end
end
    %%
    %% plot correct classsified images
if show.wrong_classified
    %Find wrigh classification
    idx3 = find(validationDS.Labels==labels);
    idx4 = int16(rand(length(idx3),1)*size(labels,1));
    fprintf('We have %d/%d correct classifications\n',length(idx3),size(labels,1));
    %%
    for jj = 150:160
        img = readimage(validationDS,idx3(jj));
        imwrite(img, 'im1.jpg');
        folder = 'F:\College\Abhishek\DCNN_NEW';
        baseFileName = 'im1.jpg';
        deconvolution(folder,baseFileName, idx3,jj, labels);
    end
        %%
        for jj = 100:110
                img = readimage(validationDS,idx4(jj));
        imwrite(img, 'im2.jpg');
        folder = 'F:\College\Abhishek\DCNN_NEW';
        baseFileName = 'im2.jpg';
        deconvolution(folder,baseFileName, idx4,jj, labels);        
        end

end
% testDS = imageDatastore('F:\College\Abhishek\DCNN_NEW\Database\','IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames'); % use imageDatastore for loading the two image categories
% [labels,err_test] = classify(XONet, testDS, 'MiniBatchSize', 60);
% confMat = confusionmat(testDS.Labels, labels);
% confMat = confMat./sum(confMat,2);
% mean(diag(confMat))








