function [AUCval, AUCtr] = plotROC(yVal, scoreVal, yTrain, scoreTrain, classNames, posCol, outPath)
%PLOTROC Plot & save ROC for train+val, return AUCs. Positive class = classNames(posCol).
    posClass = char(classNames(posCol));
    [Xv, Yv, ~, AUCval] = perfcurve(cellstr(yVal),   scoreVal(:,posCol),   posClass);
    [Xt, Yt, ~, AUCtr ] = perfcurve(cellstr(yTrain), scoreTrain(:,posCol), posClass);
    f = figure('Visible','off','Position',[100 100 700 600]);
    plot(Xv, Yv, 'b-', 'LineWidth', 2); hold on;
    plot(Xt, Yt, 'g--', 'LineWidth', 1.5);
    plot([0 1],[0 1],'k:','LineWidth',1);
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title(sprintf('ROC Curve (positive = %s)', posClass));
    legend({sprintf('Validation (AUC = %.4f)', AUCval), ...
            sprintf('Training (AUC = %.4f)', AUCtr), 'Chance'}, 'Location','southeast');
    grid on; saveas(f, outPath); close(f);
end
