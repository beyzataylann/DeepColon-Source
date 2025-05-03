import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from skimage.feature import local_binary_pattern
from tqdm import tqdm
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(in_channels = 3, num_classes = 1 )
model.load_state_dict(torch.load('/content/drive/MyDrive/bitirmepro/4unet.pth', map_location=device))
model.to(device)
model.eval()

radius = 2
n_points = 8 * radius
n_bins = n_points + 2  

root_dir = "/content/drive/MyDrive/bitirmepro/sdata"
output_dir = os.path.join(root_dir, "lbp_results/test_son1")
os.makedirs(output_dir, exist_ok=True)

img_dir = os.path.join(root_dir, "test") 
subtypes = os.listdir(img_dir)  

csv_path = os.path.join(output_dir, "lbp_features.csv")
with open(csv_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = ['filename', 'subtype'] + [f'bin_{i}' for i in range(n_bins)]
    writer.writerow(header)

    for subtype in subtypes:

        subtype_dir = os.path.join(img_dir, subtype)
        if not os.path.isdir(subtype_dir): 
            continue
        save_dir = os.path.join(output_dir, subtype)
        os.makedirs(save_dir, exist_ok=True)

        for filename in tqdm(os.listdir(subtype_dir), desc=f"Test/{subtype}"):
            if filename.lower().endswith((".jpeg", ".png", ".jpg")):
                img_path = os.path.join(subtype_dir, filename)

                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_resized = cv2.resize(image, (256, 256))

                transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5]) 
                        ])
                
                input_tensor = transform(image_resized).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    output = torch.sigmoid(output) 
                    predicted_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

                gray_image = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
                masked_image = cv2.bitwise_and(gray_image, gray_image, mask=predicted_mask)

                lbp_result = local_binary_pattern(masked_image, n_points, radius, method='default')

                hist, _ = np.histogram(lbp_result.ravel(), bins=np.arange(0, n_bins + 1), density=True)

                lbp_save_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_lbp.jpeg")
                cv2.imwrite(lbp_save_path, np.uint8(255 * lbp_result / lbp_result.max()))

                mask_save_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_mask.jpeg")
                cv2.imwrite(mask_save_path, predicted_mask * 255) 
                
                row = [filename, subtype] + hist.tolist()
                writer.writerow(row)

                print(f"LBP sonucu kaydedildi: {lbp_save_path}")
                print(f"Tahmin edilen maske kaydedildi: {mask_save_path}")