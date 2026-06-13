% introspect.m - inspect the existing trained network and dataset
clc;
fprintf('=== Loading XONet_r003.mat ===\n');
S = load('XONet_r003.mat');
fn = fieldnames(S);
fprintf('Variables in file: %s\n', strjoin(fn, ', '));
net = S.XONet;
fprintf('Network class: %s\n', class(net));

fprintf('\n=== Layers ===\n');
for i = 1:numel(net.Layers)
    L = net.Layers(i);
    fprintf('%2d  %-22s  %s\n', i, class(L), L.Name);
end

inLayer = net.Layers(1);
fprintf('\nInput size: %s\n', mat2str(inLayer.InputSize));
outLayer = net.Layers(end);
fprintf('Output classes: %s\n', strjoin(string(outLayer.Classes(:))', ', '));

fprintf('\n=== Dataset ===\n');
dbPath = fullfile(pwd, 'Database');
IMDS = imageDatastore(dbPath, 'IncludeSubfolders', true, ...
    'FileExtensions', '.jpg', 'LabelSource', 'foldernames');
fprintf('Total images: %d\n', numel(IMDS.Files));
lc = countEachLabel(IMDS);
disp(lc);
img = readimage(IMDS, 1);
fprintf('Example image size: %s\n', mat2str(size(img)));
fprintf('DONE_INTROSPECT\n');
