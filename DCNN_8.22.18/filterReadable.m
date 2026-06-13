function imdsOut = filterReadable(imds)
%FILTERREADABLE Return a copy of an imageDatastore containing only files that
% can be read as images. Unreadable/corrupt files are dropped (with a report).
    files = imds.Files;
    n = numel(files);
    good = true(n,1);
    for i = 1:n
        try
            info = imfinfo(files{i});      %#ok<NASGU>  header check (fast)
        catch
            good(i) = false;
        end
    end
    nbad = sum(~good);
    if nbad > 0
        fprintf('filterReadable: dropping %d/%d unreadable image(s):\n', nbad, n);
        badFiles = files(~good);
        for k = 1:min(nbad,20)
            fprintf('   %s\n', badFiles{k});
        end
        if nbad > 20; fprintf('   ... and %d more\n', nbad-20); end
    else
        fprintf('filterReadable: all %d images readable.\n', n);
    end
    imdsOut = subset(imds, find(good));
end
