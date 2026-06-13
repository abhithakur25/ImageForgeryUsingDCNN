clc
close all
clear all
file=dir('E:\Publication\Paper 4 Review\13\Img\*.JPG*');
for i=1:length(file)
    filename=strcat('E:\Publication\Paper 4 Review\13\Img\',file(i).name);
% [original1] = convertBinImage2RGB(filename);
[original1] = imread(filename);
cform = makecform('srgb2lab');
J = applycform(original1,cform);
%% Rotate at o degree
X = imrotate(J, 0, 'loose', 'bilinear');
 Y=imresize(X, [116 116]);
 
% % %  figure; imshow(Y)
  folder3 = 'E:\Publication\Paper 4 Review\13\Img';
  [A,map] = rgb2ind(Y,256); 
  imwrite(A,map,fullfile(folder3,sprintf('Sp_3_CI_R0-%d.jpg',i)));
% %% Rotate at 2 degree
% X = imrotate(J, 2, 'loose', 'bilinear');
%  Y=imresize(X, [114 114]);
%  
% % % %  figure; imshow(Y)
%   folder3 = 'F:\College\Abhishek\DCNN_8.21.18\Database\3\Sp';
%   [A,map] = rgb2ind(Y,256); 
%   imwrite(A,map,fullfile(folder3,sprintf('Sp_3_CI_R2-%d.jpg',i)));
%   %% Rotate at 4 degree
% X = imrotate(J, 4, 'loose', 'bilinear');
%  Y=imresize(X, [114 114]);
%  
% % % %  figure; imshow(Y)
%   folder3 = 'F:\College\Abhishek\DCNN_8.21.18\Database\3\Sp';
%   [A,map] = rgb2ind(Y,256); 
%   imwrite(A,map,fullfile(folder3,sprintf('Sp_3_CI_R4-%d.jpg',i)));
%   %% Rotate at 6 degree
% X = imrotate(J, 6, 'loose', 'bilinear');
%  Y=imresize(X, [114 114]);
%  
% % % %  figure; imshow(Y)
%   folder3 = 'F:\College\Abhishek\DCNN_8.21.18\Database\3\Sp';
%   [A,map] = rgb2ind(Y,256); 
%   imwrite(A,map,fullfile(folder3,sprintf('Sp_3_CI_R6-%d.jpg',i)));
%   %% Rotate at 8 degree
% X = imrotate(J, 8, 'loose', 'bilinear');
%  Y=imresize(X, [114 114]);
%  
% % % %  figure; imshow(Y)
%   folder3 = 'F:\College\Abhishek\DCNN_8.21.18\Database\3\Sp';
%   [A,map] = rgb2ind(Y,256); 
%   imwrite(A,map,fullfile(folder3,sprintf('Sp_3_CI_R8-%d.jpg',i)));
%   %% Rotate at 10 degree
% X = imrotate(J, 10, 'loose', 'bilinear');
%  Y=imresize(X, [114 114]);
%  
% % % %  figure; imshow(Y)
%   folder3 = 'F:\College\Abhishek\DCNN_8.21.18\Database\3\Sp';
%   [A,map] = rgb2ind(Y,256); 
%   imwrite(A,map,fullfile(folder3,sprintf('Sp_3_CI_R10-%d.jpg',i)));
end