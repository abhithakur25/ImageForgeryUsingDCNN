clc
close all
clear all
file=dir('G:\DRIVES\Thesis\image forgery1\DataBase\CASIA1\Au\*.jpg*');
for i=1:length(file)
    filename=strcat('G:\DRIVES\Thesis\image forgery1\DataBase\CASIA1\Au\',file(i).name);
    original = imread(filename);
%     original = im2double(gpuArray(imread(filename)));
    %Displaying Output%
% figure;imshow(original);
%%%%%%%%%%%%%%%%%%%%%%%
X=imresize(original, [116 116]);
 Y = imrotate(X, 10, 'loose', 'bilinear');
%  figure; imshow(Y)
  folder1 = 'F:\College\Abhishek\DCNN_OLD\Database\training_set\Au\Original\';
  [A,map] = rgb2ind(Y,256); 
  imwrite(A,map,fullfile(folder1,sprintf('Au_CASIA1_O_10-%d.jpg',i)));
%%%%%%%%%%%%%%%%%%%%%%%
load my_regioncoordinates
%Segmentation
nBins=5;
winSize=7;
nClass=6;
outImg = colImgSeg(original, nBins, winSize, nClass);
%Displaying Output%
% figure;imshow(outImg);
colormap('default');
%%%%%%%%%%%%%%%%%%%%
X=imresize(outImg, [116 116]);
 Y = imrotate(X, 10, 'loose', 'bilinear');
%  figure; imshow(Y)
  folder2 = 'F:\College\Abhishek\DCNN_OLD\Database\training_set\Au\Segmented';
%   [A,map] = rgb2ind(Y,256); 
  imwrite(Y,fullfile(folder2,sprintf('Au_CASIA1_Seg_10-%d.jpg',i)));
%%%%%%%%%%%%%%%%%%%%
cform = makecform('srgb2lab');
J = applycform(original,cform);
%%%%%%%%%%%%%%%%%
% figure;imshow(J);
%%%%%%%%%%%%%%%%%    
 X=imresize(J, [116 116]);
 Y = imrotate(X, 10, 'loose', 'bilinear');
%  figure; imshow(Y)
  folder3 = 'F:\College\Abhishek\DCNN_OLD\Database\training_set\Au\ColorIlluminate';
  [A,map] = rgb2ind(Y,256); 
  imwrite(A,map,fullfile(folder3,sprintf('Au_CASIA1_CI_10-%d.jpg',i)));

end