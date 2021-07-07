import cv2
import numpy as np
import paddle.vision.transforms as T
from PIL import Image

def img_to_tensor(img_path):

    transforms = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std =[0.229, 0.224, 0.225])
                           ])

    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_tensor = transforms(img)
    img_tensor.unsqueeze_(0)
    return img_tensor

def overlay(img_path,
            mask: np.ndarray,
            use_rgb: bool = False,
            colormap: int = cv2.COLORMAP_JET):
    
    img = cv2.imread(img_path, 1)[:, :, ::-1]
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)