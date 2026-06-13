function setupPretrained()
%SETUPPRETRAINED Make locally-cached pretrained-network support packages
% (resnet18/resnet50/alexnet/vgg16) available without the support-package-root
% machinery, by adding their package folders to the MATLAB path. The base
% toolbox functions (resnet18.m, etc.) locate their weights via the breadcrumb
% function on the path, so this is sufficient to load the pretrained networks.
    sproot = 'F:\ProgramData\MATLAB\SupportPackages\R2022b\toolbox\nnet\supportpackages';
    pkgs = {'resnet18','resnet50','alexnet','vgg16'};
    for i = 1:numel(pkgs)
        p = fullfile(sproot, pkgs{i});
        if exist(p, 'dir'); addpath(p); end
    end
end
