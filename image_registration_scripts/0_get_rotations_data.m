
data = load('C:/Users/antonio/Desktop/MsC_DataScience/MsC Thesis Project/code/3d_breast_reconstruction/data/grouped_multiplex/section_1/registered/elastic registration/save_warps/024.mat');

% Specify the parent directory containing the folders you want to loop through
parentDir = 'C:/Users/antonio/Desktop/MsC_DataScience/MsC Thesis Project/code/3d_breast_reconstruction/data/grouped_multiplex';

% Get a list of all items in the parent directory
items = dir(parentDir);

% Filter out the current and parent directory entries ('.' and '..')
folders = items([items.isdir]); % Keep only directories
folders = folders(~ismember({folders.name}, {'.', '..'})); % Exclude '.' and '..'

% Loop through each folder
for k = 1:length(folders)
    % Construct the full path to the current folder
    folderPath = fullfile(parentDir, folders(k).name, 'registered', 'elastic registration', 'save_warps');
    
    % Get a list of all files in the current folder
    files = dir(fullfile(folderPath, '*.mat'));
    
    % Loop through each file in the current folder
    for i = 1:length(files)
        % Load the current MAT file
        data = load(fullfile(folderPath, files(i).name));
        disp(['Processing file ', files(i).name]);
        
        % Check if the field 'tform' exists and has the field 'T'
         if isfield(data, 'tform')
            T_matrix = data.tform.T;
            
            % Construct the full path to the output .mat file
            outputFile = fullfile('C:/Users/antonio/Desktop/MsC_DataScience/MsC Thesis Project/code/3d_breast_reconstruction/data/warps', [files(i).name]);
            
            % Save the matrix to a .mat file
            save(outputFile, 'T_matrix');
        else
            disp('The field ''tform'' or ''T'' does not exist in the loaded .mat file.');
        end
    end
end
