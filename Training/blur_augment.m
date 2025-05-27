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
blurAmount = 15;  % Set the blur radius (odd number for Gaussian)

for k = 1:352
    oldName = allFiles(k).name;
    [~, ~, ext] = fileparts(oldName);
    
    % Read the image
    img = imread(oldName);
    
    % Apply Gaussian blur
    imgBlurred = imgaussfilt(img, blurAmount);  % Apply Gaussian filter
    
    % Save the blurred image with the new name
    newName = sprintf('%d%s', nextIndex, ext);
    imwrite(imgBlurred, newName);
    
    nextIndex = nextIndex + 1;
end

disp('Blur augmentation complete!');
