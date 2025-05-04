import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

googlenet = models.googlenet(pretrained=True)
googlenet = googlenet.to(device)
googlenet.eval()

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-2])  

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

feature_extractor = FeatureExtractor(googlenet).to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

data_dirs = {
    'train': '/content/drive/MyDrive/bitirmeprojesi/sdata/train',
    'validation': '/content/drive/MyDrive/bitirmeprojesi/sdata/validation'
}
class_names = ['colon_aca', 'colon_n']
label_map = {'colon_aca': 1, 'colon_n': 0}  

def extract_features(data_type):
    features = []
    labels = []

    for class_name in class_names:
        class_dir = os.path.join(data_dirs[data_type], class_name)
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(class_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = feature_extractor(image)
                feature_np = feature.cpu().numpy().squeeze()
                features.append(feature_np)
                labels.append(label_map[class_name])

    return np.array(features), np.array(labels)

save_dir = '/content/drive/MyDrive/bitirmeprojesi/sdata/googlenet'
os.makedirs(save_dir, exist_ok=True)  

for split in ['train', 'validation']:
    print(f"{split} verisi işleniyor...")

    features = []
    labels = []
    image_names = []
    class_list = []

    for class_name in class_names:
        class_dir = os.path.join(data_dirs[split], class_name)
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(class_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = feature_extractor(image)
                feature_np = feature.cpu().numpy().squeeze()
                features.append(feature_np)
                labels.append(label_map[class_name])
                image_names.append(img_name)
                class_list.append(class_name)

    features = np.array(features)
    labels = np.array(labels)

    
    data = np.concatenate([features, labels.reshape(-1, 1)], axis=1)

    
    feature_columns = [f'feature_{i}' for i in range(features.shape[1])]
    columns = ['image_name', 'class_name'] + feature_columns + ['label']

   
    df = pd.DataFrame(data, columns=feature_columns + ['label'])
    df.insert(0, 'class_name', class_list)
    df.insert(0, 'image_name', image_names)

    csv_path = os.path.join(save_dir, f'{split}_gfeatures.csv')
    df.to_csv(csv_path, index=False)

    print(f"{csv_path} dosyası kaydedildi.")


