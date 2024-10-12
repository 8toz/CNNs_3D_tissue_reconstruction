
% Creates a global path
%pth='C:/Users/antonio/Desktop/MsC_DataScience/MsC Thesis Project/code/3d_breast_reconstruction/data/grouped_multiplex';

% Image registration

% Calc style can be 1 or 2 choose depending on the results you get as a mask
% The registration code will by default use method 1 to create the tissue masks. 
% But it is best practice to first calculate the tissue masks and validate that 
% they are accurate to ensure the best registration quality is achieved.

% calc_style=1;
% calculate_tissue_ws(pth, calctform_style);
% 
% % Computes image registration
% calculate_image_registration(pth);

%If the elastically registered images are too jiggly, 
%   try reducing szE and/or diE inside calculate_image_registration

%If the elastically registered images are too smeared, 
%   try increasing szE and/or diE inside calculate_image_registration

%If the registration is taking too long for one image (>5 min), 
%   try reducing the resolution of the images, 
%   reduce the szE or diE inside calculate_image_registration and/or 
%   try a computer with higher RAM

% Specify the parent directory containing the folders you want to loop through
parentDir = 'C:/Users/antonio/Desktop/MsC_DataScience/MsC Thesis Project/code/3d_breast_reconstruction/data/grouped_multiplex';

% Get a list of all items in the parent directory
items = dir(parentDir);

% Filter out the current and parent directory entries ('.' and '..')
folders = items([items.isdir]); % Keep only directories
folders = folders(~ismember({folders.name}, {'.', '..'})); % Exclude '.' and '..'

% Loop through each folder
for k = 1:length(folders)
    folderName = folders(k).name;
    folderPath = fullfile(parentDir, folderName);
    disp(['Processing folder: ', folderName]);
    calc_style=1;
    calculate_tissue_ws(folderPath, calc_style);
    calculate_image_registration(folderPath);
    
    
end

