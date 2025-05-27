folderPath = 'E:\YOLO_ROI\YOLO';  % <-- Change this to your folder path
cd(folderPath);

% Supported image extensions
extensions = {'*.jpg', '*.jpeg', '*.png'};

% Collect all matching files
allFiles = [];
for i = 1:length(extensions)
    allFiles = [allFiles; dir(fullfile(folderPath, extensions{i}))];
end

% Detect the largest existing file index
maxIndex = 0;
for i = 1:length(allFiles)
    [nameOnly, ~] = strtok(allFiles(i).name, '.'); % get numeric part
    num = str2double(nameOnly);
    if ~isnan(num)
        maxIndex = max(maxIndex, num);
    end
end

% Start augmenting and saving with new names
nextIndex = maxIndex + 1;
gamma = 0.5;  % Change gamma value if needed (e.g., <1 brightens, >1 darkens)

for k = 1:352
    oldName = allFiles(k).name;
    [~, ~, ext] = fileparts(oldName);
    
    % Read and gamma-correct
    img = imread(oldName);
    img = im2double(img);  % Convert to [0,1] for gamma correction
    gammaCorrected = img .^ gamma;
    
    % Convert back to uint8 and save
    gammaCorrected = im2uint8(gammaCorrected);
    newName = sprintf('%d%s', nextIndex, ext);
    imwrite(gammaCorrected, newName);
    
    nextIndex = nextIndex + 1;
end

disp('Gamma augmentation complete!');
