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
contrastFactor = 2;  % Contrast factor (>1 for enhanced contrast, <1 for reduced contrast)

for k = 1:length(allFiles)
    oldName = allFiles(k).name;
    [~, ~, ext] = fileparts(oldName);
    
    % Read the image
    img = imread(oldName);
    
    % Apply contrast enhancement using imadjust
    % Adjust contrast: map input intensity range [0, 1] to a new range
    imgContrast = imadjust(img, stretchlim(img, [0.01 0.99]), []);  % Stretch contrast

    % Save the contrast-enhanced image with the new name
    newName = sprintf('%d%s', nextIndex, ext);
    imwrite(imgContrast, newName);
    
    nextIndex = nextIndex + 1;
end

disp('Contrast augmentation complete!');
