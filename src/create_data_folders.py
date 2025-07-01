import os
import cv2
import shutil

from tqdm.auto import tqdm

splits = ['train', 'val']
DST_DIR = '../input/split_data'
IMG_DIR = os.path.join(DST_DIR, 'images')
LBL_DIR = os.path.join(DST_DIR, 'labels')

ROOT_IMG_DIR = '../input/images'
ROOT_LBL_DIR = '../input/labels'

train_split_file = '../input/splits/train.txt'
test_split_file = '../input/splits/test.txt'

os.makedirs(DST_DIR, exist_ok=True)
for split in splits:
    os.makedirs(os.path.join(IMG_DIR, split), exist_ok=True)
for split in splits:
    os.makedirs(os.path.join(LBL_DIR, split), exist_ok=True)

with open(train_split_file, 'r') as train_file:
    train_images = train_file.readlines()
    train_images = [train_image.rstrip().split('images/')[-1] for train_image in train_images]
    print(train_images)

with open(test_split_file, 'r') as test_file:
    val_images = test_file.readlines()
    val_images = [val_image.rstrip().split('images/')[-1] for val_image in val_images]
    print(val_images)

'''
Another way to initialize the train and validation data splits
Can be adapted to different datasets

ROOT_IMG_DIR = '../input/segmentation/training/image_02'
ROOT_LBL_DIR = '../input/segmentation/training/semantic'

data_size = 200
train_size = int(0.8 * data_size)

images_dir = os.listdir(ROOT_IMG_DIR)
lables_dir = os.listdir(ROOT_LBL_DIR)
print(images_dir)

train_images = images_dir[:train_size]
train_labels = lables_dir[:train_size]

val_images = images_dir[train_size:]
val_labels = lables_dir[train_size:]
'''

def copy_data(split='train', data_list=None):
    '''
    Copy the data to the destination folder
    If direct access to the image and mask directory for split is desired,
    Uncomment the lines above and modify the path to the data

    N.B. Remove the data_name and the .png extension if the images and masks are used directly
    '''
    for data in tqdm(data_list, total=len(data_list)):
        data_name = data.split('.pcd')[0]
        image_path = os.path.join(ROOT_IMG_DIR, data_name + '.png')
        label_path = os.path.join(ROOT_LBL_DIR, data_name + '.png')

        # Since each image in the dataset used is 1241 pixels wide and 370 pixels high,
        # the image and mask will be split into two doubling the training data size
        # There will be an overlap in the image and labels since they were not divided
        # from the middle (not divided by 2) to ensure that each side of the image contains
        # the entire road (similar to a stereo camera)
        if split == 'train':
            image = cv2.imread(image_path)
            label = cv2.imread(label_path)

            image_left = image[:image.shape[0], :int(image.shape[1]//1.5)]
            image_right = image[:image.shape[0], image.shape[1]-int(image.shape[1]//1.5):image.shape[1]]

            label_left = label[:label.shape[0], :int(label.shape[1]//1.5)]
            label_right = label[:label.shape[0], label.shape[1]-int(label.shape[1]//1.5):label.shape[1]]

            cv2.imwrite(
                os.path.join(IMG_DIR, split, data_name + '_left.png'),
                image_left)
            cv2.imwrite(
                os.path.join(IMG_DIR, split, data_name + '_right.png'),
                image_right)
            
            cv2.imwrite(
                os.path.join(LBL_DIR, split, data_name + '_left.png'),
                label_left)
            cv2.imwrite(
                os.path.join(LBL_DIR, split, data_name + '_right.png'),
                label_right)
            
        else:
            shutil.copy(
                image_path,
                os.path.join(IMG_DIR, split, data_name + '.png')
            )
            shutil.copy(
                label_path,
                os.path.join(LBL_DIR, split, data_name + '.png')
            )

copy_data(split='train', data_list=train_images)
copy_data(split='val', data_list=val_images)