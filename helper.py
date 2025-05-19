import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

input_nc = 3
output_nc = 3

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        weights = self.fc(x)
        return x * weights

# Residual Block with SE
class ResnetBlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)
        self.se = SEBlock(dim, reduction)
    
    def build_conv_block(self, dim):
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim)
        ]
        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        out = self.conv_block(x)
        out = self.se(out)
        return x + out

# Generator Network
class GeneratorResNet(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(GeneratorResNet, self).__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResnetBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# Instantiate models
netG_A2B = GeneratorResNet(input_nc, output_nc)
netG_B2A = GeneratorResNet(input_nc, output_nc)

# Load model weights
device = 'cpu'
netG_A2B.load_state_dict(torch.load('./netG_A2B_epoch130.pth', map_location=device))
netG_B2A.load_state_dict(torch.load('./netG_B2A_epoch130.pth', map_location=device))

# Image transformation functions
def generate_Y2O(uploaded_image):
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(uploaded_image).unsqueeze(0)  # Add batch dimension
    old = netG_A2B(tensor)
    return (old.squeeze().detach().permute(1, 2, 0).numpy() + 1) / 2

def generate_O2Y(uploaded_image):
    img = cv2.resize(uploaded_image, (256, 256))
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(img).unsqueeze(0)  # Add batch dimension
    young = netG_B2A(tensor)
    return (young.squeeze().detach().permute(1, 2, 0).numpy() + 1) / 2

# Face detection using OpenCV
def extract_faces_opencv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    face_crops = []
    for (x, y, w, h) in faces:
        y1, y2 = max(0, y - 50), min(image.shape[0], y + h)
        x1, x2 = max(0, x - 30), min(image.shape[1], x + w + 30)
        face_crop = image[y1:y2, x1:x2]
        face_crops.append(face_crop)
    return face_crops
