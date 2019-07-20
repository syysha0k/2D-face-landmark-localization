import torch
import torch.nn as nn
from core.conv_warpper import ConvWarpper

class DarkNet(nn.Module):
	def __init__(self):
		super().__init__()
		Conv = ConvWarpper
		self.conv1 = Conv(3, 16, 3, padding=1, relu=True, batch_norm=False)
		self.pool1 = nn.MaxPool2d(2, 2)
		self.conv2 = Conv(16, 32, 3, padding=1, relu=True, batch_norm=True)
		self.pool2 = nn.Maxpool2d(2, 2)
		self.conv3_1s = Conv(32, 16, 1, padding=0, relu=False, batch_norm=False)
		self.conv3_1 = Conv(16, 128, 3, padding=1, relu=True, batch_norm=False)
		self.conv3_2s = Conv(128, 16, 1, padding=0, relu=False, batch_norm=False)
		self.conv3_2 = Conv(16, 128, 3, padding=1, relu=True, batch_norm=True)
		self.pool3 = nn.Maxpool2d(2, 2)
		self.conv4_1s = Conv(128, 32, 1, padding=0, relu=False, batch_norm=False)
		self.conv4_1 = Conv(32, 256, 3, padding=1, relu=True, batch_norm=False)
		self.conv4_2s = Conv(256, 32, 1, padding=0, relu=False, batch_norm=False)
		self.conv4_2 = Conv(32, 256, 3, padding=1, relu=True, batch_norm=True)
		self.pool4 = nn.Maxpool2d(2, 2)
		self.conv5_1s = Conv(256, 64, 1, padding=0, relu=False, batch_norm=False)
		self.conv5_1 = Conv(64, 512, 3, padding=1, relu=True, batch_norm=False)
        self.conv5_2s = Conv(512, 64, 1, padding=0, relu=False, batch_norm=False)
		self.conv5_2 = Conv(64, 512, 3, padding=1, relu=True, batch_norm=True)
		self.avgpool = nn.AvgPool2d(3, stride=1)
		self.fc = nn.Linear(512, 2)
	
	def forward(self, x):
		out = self.conv1(x)
		out = self.pool1(out)
		out = self.conv2(out)
		out = self.pool2(out)
		out = self.conv3_1s(out)
		out = self.conv3_1(out)
		out = self.conv3_2s(out)
		out = self.conv3_2(out)
		out = self.pool3(out)
		out = self.conv4_1s(out)
		out = self.conv4_1(out)
		out = self.conv4_2s(out)
		out = self.conv4_2(out)
		out = self.pool4(out)
		out = self.conv5_1s(out)
		out = self.conv5_1(out)
		out = self.conv5_2s(out)
		out = self.conv5_2(out)
		out = self.avgpool(x)
		out = self.fc(x)
		return out
	

















