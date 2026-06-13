function printMetrics(tag, m)
%PRINTMETRICS Pretty-print a metrics struct from computeMetrics.
    fprintf('\n----- %s -----\n', tag);
    fprintf('Accuracy: %.4f\n', m.accuracy);
    fprintf('%-10s %10s %10s %10s %8s\n','Class','Precision','Recall','F1','Support');
    for i = 1:numel(m.classNames)
        fprintf('%-10s %10.4f %10.4f %10.4f %8d\n', m.classNames(i), ...
            m.precision(i), m.recall(i), m.f1(i), m.support(i));
    end
    fprintf('%-10s %10.4f %10.4f %10.4f\n','Macro',   m.macroPrecision, m.macroRecall, m.macroF1);
    fprintf('%-10s %10.4f %10.4f %10.4f\n','Weighted',m.weightedPrecision, m.weightedRecall, m.weightedF1);
end
