% --- Main script code here ---

folder1 = 'E:\Hg\ambient';
folder2 = 'E:\Hg\box';

[conc_ambient, change_ambient] = process_folder(folder1);
[conc_box, change_box] = process_folder(folder2);

figure;
semilogx(conc_ambient, change_ambient, '-o', 'LineWidth', 2, 'DisplayName', 'Ambient');
hold on;
semilogx(conc_box, change_box, '-s', 'LineWidth', 2, 'DisplayName', 'Box');
xlabel('Concentration');
ylabel('Average % Change in RGB vs. 0 Concentration');
title('Average RGB % Change vs. Concentration');
legend('Location', 'best');
grid on;
hold off;

% --- Put your function here, at the end of the script ---

function [concentrations, avg_abs_percent_changes] = process_folder(folder)
    % Get all jpg and png images
    image_files = [dir(fullfile(folder, '*.jpg')); dir(fullfile(folder, '*.png'))];

    % Find zero concentration image (has '0' in name)
    zero_idx = find(contains({image_files.name}, '0'), 1);
    if isempty(zero_idx)
        error(['No zero concentration reference image found in folder: ', folder]);
    end
    zero_filename = image_files(zero_idx).name;

    % Read zero image and average RGB
    ref_img = imread(fullfile(folder, zero_filename));
    ref_avg_R = mean(ref_img(:,:,1), 'all');
    ref_avg_G = mean(ref_img(:,:,2), 'all');
    ref_avg_B = mean(ref_img(:,:,3), 'all');

    concentrations = [];
    avg_abs_percent_changes = [];

    for i = 1:length(image_files)
        img_name = image_files(i).name;
        if strcmp(img_name, zero_filename)
            continue;
        end

        conc_str = regexp(img_name, '(\d+)', 'match');
        if isempty(conc_str)
            continue;
        end
        concentration = str2double(conc_str{1});

        img = imread(fullfile(folder, img_name));
        avg_R = mean(img(:,:,1), 'all');
        avg_G = mean(img(:,:,2), 'all');
        avg_B = mean(img(:,:,3), 'all');

        abs_percent_change_R = abs(avg_R - ref_avg_R) / 255 * 100;
        abs_percent_change_G = abs(avg_G - ref_avg_G) / 255 * 100;
        abs_percent_change_B = abs(avg_B - ref_avg_B) / 255 * 100;
        avg_abs_percent_change = mean([abs_percent_change_R, abs_percent_change_G, abs_percent_change_B]);

        concentrations(end+1) = concentration;
        avg_abs_percent_changes(end+1) = avg_abs_percent_change;
    end

    % Sort results
    [concentrations, sort_idx] = sort(concentrations);
    avg_abs_percent_changes = avg_abs_percent_changes(sort_idx);
end
