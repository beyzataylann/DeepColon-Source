import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet18(pretrained=True).to(device)
resnet.eval()

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])  

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

feature_extractor = FeatureExtractor(resnet).to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

data_dirs = {
    'train': '/content/drive/MyDrive/bitirmeprojesi/sdata/train',
    'validation': '/content/drive/MyDrive/bitirmeprojesi/sdata/validation',
    
}
class_names = ['colon_aca', 'colon_n']
label_map = {'colon_aca': 1, 'colon_n': 0}
save_dir = '/content/drive/MyDrive/bitirmeprojesi/sdata/resnet18'
os.makedirs(save_dir, exist_ok=True)

def extract_and_save(split):
    print(f"{split} verisi işleniyor...")
    features = []
    labels = []
    filenames = []
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
            filenames.append(img_name)
            class_list.append(class_name)

    feature_cols = [f'feature_{i}' for i in range(len(features[0]))]
    df = pd.DataFrame(features, columns=feature_cols)
    df.insert(0, 'class_name', class_list)
    df.insert(0, 'image_name', filenames)
    df['label'] = labels

    csv_path = os.path.join(save_dir, f'{split}_rfeatures.csv')
    df.to_csv(csv_path, index=False)
    print(f"{csv_path} kaydedildi.")

for split in ['train', 'validation']:
    extract_and_save(split)
