function m = computeMetrics(yTrue, yPred, classNames)
%COMPUTEMETRICS Per-class & aggregate precision/recall/F1/accuracy.
    order = cellstr(classNames);
    yTrue = cellstr(string(yTrue));
    yPred = cellstr(string(yPred));
    C = confusionmat(yTrue, yPred, 'Order', order);
    n = sum(C(:));
    K = numel(classNames);
    precision = zeros(K,1); recall = zeros(K,1); f1 = zeros(K,1); support = zeros(K,1);
    for i = 1:K
        TP = C(i,i);
        FP = sum(C(:,i)) - TP;
        FN = sum(C(i,:)) - TP;
        support(i) = sum(C(i,:));
        precision(i) = safediv(TP, TP+FP);
        recall(i)    = safediv(TP, TP+FN);
        f1(i)        = safediv(2*precision(i)*recall(i), precision(i)+recall(i));
    end
    m.classNames = classNames;
    m.confmat    = C;
    m.accuracy   = sum(diag(C)) / n;
    m.precision  = precision;
    m.recall     = recall;
    m.f1         = f1;
    m.support    = support;
    m.macroPrecision = mean(precision);
    m.macroRecall    = mean(recall);
    m.macroF1        = mean(f1);
    w = support / n;
    m.weightedPrecision = sum(w .* precision);
    m.weightedRecall    = sum(w .* recall);
    m.weightedF1        = sum(w .* f1);
end

function v = safediv(a,b)
    if b == 0; v = 0; else; v = a/b; end
end
