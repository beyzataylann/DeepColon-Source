#CLAHE VE ORTALAMA FİLTRELEME
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Örnek bir görüntüyü oku
image_path = "/content/drive/MyDrive/bitirmepro/train/colon_aca/colonca1.jpeg"
original_image = cv2.imread(image_path)

lab_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab_image)

# CLAHE uygula
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
L_clahe = CLAHE.apply(L)

lab_image_clahe = cv2.merge([L_clahe, A, B])
enhanced_image = cv2.cvtColor(lab_image_clahe, cv2.COLOR_LAB2BGR)

# Ortalama filtreleme uygula
blurred_image = cv2.blur(enhanced_image, (3,3))

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title("Orijinal Görüntü")

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
plt.title("CLAHE Uygulanmış Görüntü")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.title("CLAHE + Ortalama Filtreleme")

plt.show()


def plot_histogram(image, title, color):
    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    plt.plot(hist, color=color)
    plt.title(title)
    plt.xlim([0,256])

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plot_histogram(L, "Orijinal L Kanalı Histogramı", 'blue')

plt.subplot(1,3,2)
plot_histogram(L_clahe, "CLAHE Uygulanmış L Kanalı Histogramı", 'red')

plt.subplot(1,3,3)
plot_histogram(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY), "CLAHE + Ortalama Filtreleme Histogramı", 'green')

plt.show()

