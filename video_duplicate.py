import os
import shutil

# Constants
source_file = 'IMG_4475.MOV'
target_dir = 'video_data'
num_copies = 200

# Create target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Copy and rename the file
for i in range(num_copies):
    # Format the new filename
    new_filename = f"IMG_{i:04d}.MOV"
    new_filepath = os.path.join(target_dir, new_filename)

    # Copy the file
    shutil.copy2(source_file, new_filepath)

print(f"Finished duplicating {source_file} {num_copies} times in '{target_dir}' directory.")
