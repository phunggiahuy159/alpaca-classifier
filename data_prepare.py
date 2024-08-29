import os
import shutil
from sklearn.model_selection import train_test_split



data_dir = 'dataset'
train_dir = 'train'
test_dir = 'test'
 
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
# Categories
categories = ['alpaca', 'not alpaca']
for category in categories:
    # Create subdirectories for train and test
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)
    # Get list of all files in the category
    category_path = os.path.join(data_dir, category)
    files = os.listdir(category_path)
    # Split the files into train and test
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    # Copy files to train directory
    for file in train_files:
        src = os.path.join(category_path, file)
        dst = os.path.join(train_dir, category, file)
        shutil.copyfile(src, dst)
    # Copy files to test directory
    for file in test_files:
        src = os.path.join(category_path, file)
        dst = os.path.join(test_dir, category, file)
        shutil.copyfile(src, dst)
