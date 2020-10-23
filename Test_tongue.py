import torch
import numpy as np
import cv2
from efficientnet_pytorch import EfficientNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pickle

# params below must be the same as training's
model_name = 'efficientnet-b2'
num_classes = 24
image_size = 768

# load from training checkpoint
ckpt = torch.load('/Users/forever/PycharmProjects/Leo/Tongue_class/outputs/model_epoch[78]_metric[0.6930].pth',
                  map_location='cpu')
model = EfficientNet.from_name(model_name, num_classes=24, image_size=image_size)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# load image
img_path = '/Users/forever/darknet/tongue_test/111.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# transform to tensor
input = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225], ),
    ToTensorV2()
])(image=img)['image'].unsqueeze(0)  # input tensor should be 4 dimensions

with torch.no_grad():
    out = model(input)
torch.sigmoid(out)

outs = torch.sigmoid(out).squeeze(0)
conf = 0.6     # threshold
test_list = [1 if x > conf else 0 for x in outs]
print(test_list)

d = pickle.load(open("/Users/forever/Desktop/label_dict", "rb"))
index = 0
for i in test_list:
    if i:
        print({v: k for k, v in d.items()}.get(index))
    index += 1
