close all; clear all; clear variables; clear global; clc;    % clean desk
% A trained network is loaded from disk to save time when running the
% example. Set this flag to true to train the network.
for x=1:1;

doTraining              = true;

% load the file data for training the CNN
% addpath='H:\TU\thesis\thesis sem4\video databases\DS'
 train1 = imageDatastore('F:\College\Abhishek\DCNN_NEW\Database\','IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames'); % use imageDatastore for loading the two image categories
% train1 = imageDatastore('F:\College\Harpreet Kaur\dataset\database-test','IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames'); % use imageDatastore for loading the two image categories
example_image = readimage(train1,1);                      % read one example image
numChannels = size(example_image,3);                    % get color information
numImageCategories = size(categories(train1.Labels),1);   % get category labels
[trainingDS,validationDS] = splitEachLabel(train1,0.8); % generate training and validation set
LabelCnt = countEachLabel(train1);                        % load lable information
for m=1:numImageCategories                           % print out how many images we have for each category
    fprintf('%s\t%d\n',LabelCnt.Label(m),LabelCnt.Count(m));
end

%%
if doTraining
    %% Setup of the CNN
    % Convolutional layer parameters
    filterSize = [5 5]
    numFilters = 32
    inputLayer = imageInputLayer(size(example_image));  % input layer with no data augmentation
    
    

    middleLayers = [
        % The first convolutional layer has a bank of numFilters filters of size filterSize. A
        % symmetric padding of 4 pixels is added.
        convolution2dLayer(filterSize, numFilters, 'Padding', 4)
        % Next add the ReLU layer:
        reluLayer()
        % Follow it with a max pooling layer that has a 5x5 spatial pooling area
        % and a stride of 2 pixels. This down-samples the data dimensions.
        maxPooling2dLayer(5, 'Stride', 2)
        % % % Repeat the 3 core layers to complete the middle of the network.
        convolution2dLayer(filterSize, numFilters*2, 'Padding', 4)
        reluLayer()
        maxPooling2dLayer(5, 'Stride',2)
        convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2)
        reluLayer()
        maxPooling2dLayer(3, 'Stride',2)];
    convolution2dLayer(3,32)
          reluLayer
          maxPooling2dLayer(3,'Stride',2)
          
          
          convolution2dLayer(3,64)
          reluLayer
          maxPooling2dLayer(3,'Stride',2)
          
          
          convolution2dLayer(3,64)
          reluLayer
          maxPooling2dLayer(3,'Stride',2)       
    
    finalLayers = [
        % % Add a fully connected layer with 2 output neurons.
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
    
    % Set the network training options
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
    % 'MiniBatchSize' reduced from 128 to 64 because GPU ran out of memory
        % for Quadro GPU
    % 'MiniBatchSize' increased from 128 to >512 -> TitanX GPU but than
        % more example images would be needed (too much averaging)
    
    % Train a network.
    training = trainNetwork(trainingDS, layers, opts);
    save('training_002.mat','training');
    load('training_002.mat');
end
load('training_002.mat');
% load('validationDS.mat');
%% test the performance
% test network performance on validation set
save('validationDS.mat','validationDS');
[labels,~] = classify(training, validationDS, 'MiniBatchSize', 128);
confMat = confusionmat(validationDS.Labels, labels);
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
fprintf('Performance on validation set \t\t\t%.4f\n',mean(diag(confMat)));
  
%% plot true classsified 
    % % Find classified examples
    idx = find(validationDS.Labels==labels);
    idx2 = int16(rand(length(idx),1)*size(labels,1));
    fprintf('We have %d/%d true classifications\n',length(idx),size(labels,1));
    for jj = 1:length(idx)
        img = readimage(validationDS,idx(jj));
        imwrite(img, 'im1.jpg');
        im1=imread('im1.jpg');
        figure(1);
        subplot(1,2,1);
%         imagesc(im1);
        imshow(im1);
        lab = sprintf('classified as %s',labels(idx(jj)));
        title(lab,'Color','Red');
        img = double(readimage(validationDS,idx2(jj)));
        imwrite(img, 'im2.jpg');
        im2=imread('im2.jpg');
        figure(1);
         subplot(1,2,2);
%         imagesc(im2);
        imshow(im2)
        lab = sprintf('classified as %s',labels(idx2(jj)));
        title(lab,'Color','Green');
saveas(figure(jj),fullfile('F:\College\Abhishek\DCNN_NEW\Results',['figure' num2str(jj) '.jpeg']));
    end
end
% end