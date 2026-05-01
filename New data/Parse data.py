import os

rootdir = 'C:/Users/Rudko/Documents/GitHub/Spheroids/New data'

for subdir, dirs, files in os.walk(rootdir):
    for f in files:
        
        filename, file_extension = os.path.splitext(f)
        
        if(file_extension != '.csv'):
            continue
        
        elements = filename.split('-')
        
        new_name = elements[2] + '_' + elements[0] + '_' + elements[1]
        
        os.rename(os.path.join(subdir, f), os.path.join(subdir, new_name + file_extension))