import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder path
        self.enc1 = DoubleConv(1, 64)  # 1 channel for grayscale input
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder path
        self.up_conv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 1024 due to skip connection
        
        self.up_conv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.up_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)  # 1 channel output for binary mask
        self.sigmoid = nn.Sigmoid()
        
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.up_conv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.up_conv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.up_conv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.up_conv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        x = self.final_conv(x)
        return self.sigmoid(x)

def get_model():
    model = UNet()
    return model
