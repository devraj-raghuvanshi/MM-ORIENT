!pip install keras_ocr

import keras_ocr
import cv2
import math
import numpy as np
import os
from PIL import Image
from google.colab.patches import cv2_imshow

#@title Run this cell for setup { display-mode: "form"}
!git clone https://github.com/vrindaprabhu/deepfillv2_colab.git
!gdown "https://drive.google.com/u/0/uc?id=1uMghKl883-9hDLhSiI8lRbHCzCmmRwV-&export=download"
!mv /content/deepfillv2_WGAN_G_epoch40_batchsize4.pth deepfillv2_colab/model/deepfillv2_WGAN.pth

cd deepfillv2_colab

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

def create_mask(img_path):
    # read image
    img = keras_ocr.tools.read(img_path)
    pipeline= keras_ocr.pipeline.Pipeline()
    prediction_groups = pipeline.recognize([img])

    mask = np.zeros(img.shape[:2], dtype="uint8")

    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))

        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,
                 thickness)

    return(mask)

!pip install opencv-python==4.1.2

def mask_input(file):
  i = cv2.imread(file)
  output = create_mask(i)
  output = Image.fromarray(output)
  output.save("/mask_wis/" + os.path.basename(file))

done_files = []
for filename in os.listdir("/mask_wis/"):
  done_files.append(filename)

not_done_files = []

import os
def mask_all_input(input_folder):
  for filename in os.listdir(input_folder):
    f = os.path.join(input_folder, filename)
    # print(f)
    if((f[len(f)-1]=='g' and f[len(f)-2]=='p' and f[len(f)-3]=='j' and f[len(f)-4]=='.') or (f[len(f)-1]=='g' and f[len(f)-2]=='n' and f[len(f)-3]=='p' and f[len(f)-4]=='.') and filename not in done_files):
      print(f)
      try:
        mask_input(f)
      except:
        print("ERROR")
        not_done_files.append(filename)
        continue
      done_files.append(filename)
    #mask_input(f)
mask_all_input("/wis")

#@title Run to trigger inpainting. { display-mode: "form" }
!python inpaint.py