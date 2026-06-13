function saveConfusion(yTrue, yPred, ttl, outPath)
%SAVECONFUSION Save a confusion chart (row- and column-normalized) to PNG.
    f = figure('Visible','off','Position',[100 100 650 600]);
    cc = confusionchart(yTrue, yPred);
    cc.Title = ttl;
    cc.RowSummary = 'row-normalized';
    cc.ColumnSummary = 'column-normalized';
    saveas(f, outPath); close(f);
end
