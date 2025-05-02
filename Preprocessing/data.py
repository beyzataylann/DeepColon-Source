import os
import shutil
import random

from sklearn.model_selection import train_test_split


train_ratio = 0.7      
validation_ratio = 0.15  
test_ratio = 0.15     


dataset_dir = "/content/drive/MyDrive/bitirmepro/colon_image_sets"  
output_dir = "/content/drive/MyDrive/bitirmepro"  

train_dir = os.path.join(output_dir, "train")
validation_dir = os.path.join(output_dir, "validation")
test_dir = os.path.join(output_dir, "test")

for folder in [train_dir, validation_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

classes = os.listdir(dataset_dir)

for class_name in classes:
    class_path = os.path.join(dataset_dir, class_name)  

    if not os.path.isdir(class_path):
        continue

    
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)


    images = os.listdir(class_path)

    random.shuffle(images)

    train_images, temp_images = train_test_split(images, test_size=(validation_ratio + test_ratio))
    validation_images, test_images = train_test_split(temp_images, test_size=test_ratio/(validation_ratio + test_ratio))


    def copy_images(image_list, source_folder, destination_folder):
        for image in image_list:
            source_path = os.path.join(source_folder, image)
            destination_path = os.path.join(destination_folder, image)
            shutil.copy(source_path, destination_path)

    copy_images(train_images, class_path, os.path.join(train_dir, class_name))
    copy_images(validation_images, class_path, os.path.join(validation_dir, class_name))
    copy_images(test_images, class_path, os.path.join(test_dir, class_name))

print("Veri başarıyla train, validation ve test kümelerine ayrıldı!")
