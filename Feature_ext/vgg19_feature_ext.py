import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_names = ['colon_aca', 'colon_n']
label_map = {'colon_aca': 1, 'colon_n': 0}

data_dirs = {
    'train': '/content/drive/MyDrive/bitirmepro/sdata/train',
    'val': '/content/drive/MyDrive/bitirmepro/sdata/validation'
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

vgg19 = torchvision.models.vgg19(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(vgg19.children())[:-1])
feature_extractor.to(device)
feature_extractor.eval()

def extract_features(data_type):
    features = []
    labels = []
    filenames = []

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
                feature_np = feature.cpu().numpy().reshape(-1)  # [1, 512, 7, 7] → [25088]
                features.append(feature_np)
                labels.append(label_map[class_name])
                filenames.append(img_name)

    feature_columns = [f'feature_{i}' for i in range(features[0].shape[0])]
    df_features = pd.DataFrame(features, columns=feature_columns)

    df_features.insert(0, 'filename', filenames)
    df_features.insert(1, 'label', labels)
    return df_features

df_train = extract_features('train')
df_val = extract_features('val')

df_train.to_csv('train2_features.csv', index=False)
df_val.to_csv('val2_features.csv', index=False)

print("Özellik çıkarımı ve CSV kaydı tamamlandı.")
