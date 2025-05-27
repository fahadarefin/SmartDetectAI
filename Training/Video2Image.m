% Define the video file path and name
video_file = 'TPE_sample.mp4'; % Replace with your video file path

% Create a VideoReader object
video = VideoReader(video_file);

% Calculate the total number of frames in the video
total_frames = video.NumFrames;

% Define the number of frames to extract (e.g., 30)
num_frames_to_extract = 50;

% Calculate the interval between frames to extract
frame_interval = floor(total_frames / num_frames_to_extract);

% Define the output folder for images
output_folder = 'E:\TPE'; % Replace with your desired output folder

% Create the output folder if it doesn't exist
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Loop to extract and save frames
for i = 1:num_frames_to_extract
    % Calculate the frame number to extract
    frame_number = (i-1) * frame_interval + 1;
    
    % Read the frame
    video.CurrentTime = (frame_number - 1) / video.FrameRate;
    frame = readFrame(video);
    
    % Define the output filename
    output_filename = sprintf('NoSalt_AGNSE@TPE (%d).jpg', i);
    
    % Save the frame as an image
    imwrite(frame, fullfile(output_folder, output_filename));
end

disp('Frames have been extracted and saved successfully.');
