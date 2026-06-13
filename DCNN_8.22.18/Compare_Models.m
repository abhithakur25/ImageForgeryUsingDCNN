%% Compare_Models.m
% Aggregate every model's saved results into one comparison table (CSV + TXT)
% and a grouped bar chart. Scans Results\*\results_*.mat.

clc;
rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir); rootDir = pwd; end
resRoot = fullfile(rootDir, 'Results');
matFiles = dir(fullfile(resRoot, '**', 'results_*.mat'));

rows = {};
for i = 1:numel(matFiles)
    fp = fullfile(matFiles(i).folder, matFiles(i).name);
    S = load(fp);
    if ~isfield(S, 'results'); continue; end
    r = S.results;
    [~, modelName] = fileparts(matFiles(i).folder);
    v = r.val; t = r.train;
    rows(end+1,:) = { modelName, ...
        t.accuracy, v.accuracy, ...
        v.macroPrecision, v.macroRecall, v.macroF1, ...
        getf(r,'AUC_val',NaN) };  %#ok<AGROW>
end

if isempty(rows)
    fprintf('No results_*.mat found under %s\n', resRoot); return;
end

T = cell2table(rows, 'VariableNames', ...
    {'Model','TrainAcc','ValAcc','ValPrecision','ValRecall','ValF1','ValAUC'});
T = sortrows(T, 'ValAcc', 'descend');

% Write CSV + pretty TXT
writetable(T, fullfile(resRoot, 'COMPARISON.csv'));
fid = fopen(fullfile(resRoot, 'COMPARISON.txt'), 'w');
fprintf(fid, 'MODEL COMPARISON (sorted by validation accuracy)\n');
fprintf(fid, 'Generated: %s\n\n', datestr(now));
fprintf(fid, '%-24s %9s %9s %10s %9s %8s %8s\n', ...
    'Model','TrainAcc','ValAcc','ValPrec','ValRecall','ValF1','ValAUC');
for i = 1:height(T)
    fprintf(fid, '%-24s %9.4f %9.4f %10.4f %9.4f %8.4f %8.4f\n', ...
        T.Model{i}, T.TrainAcc(i), T.ValAcc(i), T.ValPrecision(i), ...
        T.ValRecall(i), T.ValF1(i), T.ValAUC(i));
end
fclose(fid);
type(fullfile(resRoot, 'COMPARISON.txt'));

% Bar chart of validation accuracy / F1 / AUC
f = figure('Visible','off','Position',[100 100 1000 600]);
metrics = [T.ValAcc, T.ValF1, T.ValAUC];
bar(metrics); grid on;
set(gca, 'XTickLabel', T.Model, 'XTick', 1:height(T)); xtickangle(30);
ylabel('Score'); ylim([0 1]); legend({'Val Accuracy','Val F1','Val AUC'}, 'Location','southoutside','Orientation','horizontal');
title('Model Comparison (validation)');
saveas(f, fullfile(resRoot, 'COMPARISON_bar.png')); close(f);
fprintf('\nComparison written to %s\n', resRoot);
fprintf('DONE_COMPARE\n');

function val = getf(s, name, dflt)
    if isfield(s, name); val = s.(name); else; val = dflt; end
end
