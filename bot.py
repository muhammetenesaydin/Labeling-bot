# Bu kod betiği elimizdeki yolov7 ağırlık dosyası ile istenilen sayıda fotoğrafı etiketleme yapıyor 
# This code script labels the desired number of photos with the yolov7 weight file we have.


import os
import torch
import cv2
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords

# Model ve ağırlık dosyasını yükle
# Load model and weight file
weights_path = 'yolov7.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights_path, map_location=device)
stride = int(model.stride.max())

# Görüntüleri ve etiket klasörünü yükle
# Load images and tags folder
image_folder = 'raw data folder direction'
output_folder = 'output folder direction'
label_folder = 'label folder direction'
img_output_subfolder = 'img'
img_size = 640
dataset = LoadImages(image_folder, img_size=img_size)

# Modeli değerlendirme modunda ayarla
# Set the model in evaluation mode
model.eval()

# Çıktı klasörünü, etiket klasörünü ve img alt klasörünü oluştur
# Create output folder, tag folder and img subfolder
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, label_folder), exist_ok=True)
os.makedirs(os.path.join(output_folder, img_output_subfolder), exist_ok=True)

# Görüntüler üzerinde döngü
# Loop through images
for path, img, im0s, _ in dataset:
    filename = os.path.splitext(os.path.basename(path))[0]  # Dosya adını al

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Görüntüyü modelden geçir
    # Pass image through model

    pred = model(img)[0]

    # Non-maximum suppression (NMS) uygula
    # Apply non-maximum suppression (NMS)

    pred = non_max_suppression(pred, 0.25, 0.45)

    # Sonuçları işle
    # Process results

    with open(os.path.join(output_folder, label_folder, f'{filename}.txt'), 'w') as f:
        for j, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                for *xyxy, conf, cls in det:
                    # Koordinatları normalleştirerek kaydet
                    # Save coordinates by normalizing them
                    x_center = ((xyxy[0] + xyxy[2]) / 2) / im0s.shape[1]
                    y_center = ((xyxy[1] + xyxy[3]) / 2) / im0s.shape[0]
                    width = (xyxy[2] - xyxy[0]) / im0s.shape[1]
                    height = (xyxy[3] - xyxy[1]) / im0s.shape[0]
                    # Etiketi dosyaya yaz
                    # Write label to file
                    line = f"0 {x_center:.7f} {y_center:.7f} {width:.7f} {height:.7f}\n"
                    f.write(line)

                    # Etiketi çiz
                    # Draw the label
                    """
                    color = (0, 255, 0)  # Yeşil renk
                    thickness = 2
                    start_point = (int(xyxy[0]), int(xyxy[1]))
                    end_point = (int(xyxy[2]), int(xyxy[3]))
                    im0s = cv2.rectangle(im0s, start_point, end_point, color, thickness)
                    """

    # 640x640 boyutunda görüntüyü img klasörüne PNG formatında kaydet
    # Save the 640x640 image in PNG format to the img folder
    resized_img = cv2.resize(im0s, (640, 640))
    img_output_path = os.path.join(output_folder, img_output_subfolder, f'{filename}.jpg')
    cv2.imwrite(img_output_path, resized_img)

    #cv2.imshow('output', im0s)
    cv2.waitKey(0)

cv2.destroyAllWindows()
