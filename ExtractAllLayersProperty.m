%% Visualize Features of a Convolutional Neural Network

clc
close all
%% Load Pretrained Network
% Load a pretrained AlexNet network.
load('XONet_r005.mat')
net = XONet;
net.Layers

%%
% *Features on Convolutional Layer 1*

layer = 2;
name = net.Layers(layer).Name

%%
% Visualize the first 56 features learned by this layer using

channels = 1:32;

I = deepDreamImage(net,layer,channels, ...
    'PyramidLevels',1);

figure
montage(I)
title(['Layer ',name,' Features'])


%%
% *Features on Convolutional Layer 2*
channels = 1:30;

I = deepDreamImage(net,layer,channels,...
    'PyramidLevels',1);

figure
montage(I)
name = net.Layers(layer).Name;
title(['Layer ',name,' Features'])

%%
% *Features on Convolutional Layers 3&ndash;5*
% Convolution Layer Features
layers = [2 5 8];
channels = 1:30;

for layer = layers
    I = deepDreamImage(net,layer,channels,...
        'Verbose',false, ...
        'PyramidLevels',1);
    
    figure
    montage(I)
    name = net.Layers(layer).Name;
    title(['Layer ',name,' Features'])
end
% ReLU Layer Features
layers = [3 6 9];
channels = 1:30;

for layer = layers
    I = deepDreamImage(net,layer,channels,...
        'Verbose',false, ...
        'PyramidLevels',1);
    
    figure
    montage(I)
    name = net.Layers(layer).Name;
    title(['Layer ',name,' Features'])
end

% MaxPool Layer Features
layers = [4 7 10];
channels = 1:30;

for layer = layers
    I = deepDreamImage(net,layer,channels,...
        'Verbose',false, ...
        'PyramidLevels',1);
    
    figure
    montage(I)
    name = net.Layers(layer).Name;
    title(['Layer ',name,' Features'])
end
%% Visualize Fully Connected Layers

%%
% To produce images that resemble each class the most closely, select the
% final fully connected layer, and set |channels| to be the indices of the
% classes.
layer = [11 13];
channels = 32;
net.Layers(end).ClassNames(channels)

% %%
% % Generate detailed images that strongly activate these classes.
I = deepDreamImage(net,layer,channels, ...
    'Verbose',false, ...
    'NumIterations',50);

figure
montage(I)
name = net.Layers(layer).Name;
title(['Layer ',name,' Features'])
