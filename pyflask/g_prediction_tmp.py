import numpy as np
import time
import os
import os.path as osp

import numpy as np 
#import pandas as pd 
import pickle
import torch
from PIL import Image
from PIL import ImageEnhance
import random
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision import models, transforms
from torch.autograd import Variable
from torch import topk
import copy
import click
import matplotlib.cm as cm
from torchvision import models, transforms

import cv2
import pdb
from collections import Sequence

import torch as t

from cv2 import imshow as cv2_imshow
import matplotlib.pyplot as plt

from . import GoogLeNet
from . import GoogLeNetBinary


print(os.getcwd())

def preprocess_image(image_path):
	'''입력받은 이미지 전처리 후 4차원의 텐서타입으로 변경'''
	#img = load_img(image_path, target_size=(img_height, img_width)) # (400, 381)
	'''The img_to_array() function adds channels: x.shape = (224, 224, 3) for RGB and (224, 224, 1) for gray image'''
	img = Image.open(image_path).convert('L')    
	width = img.size[0]
	height = img.size[1]
	croppedImage=img.crop((width*0.11, height*0.11, (width*0.89), (height*0.89))) 
	original_width, original_height = img.size
	
	resize_img = croppedImage.resize((224, 224))
	#img_enhance = ImageEnhance.Contrast(resize_img).enhance(2)
	re_img_arr=np.asarray(resize_img)
	arr = (re_img_arr - re_img_arr.min()) / (re_img_arr.max() - re_img_arr.min())
	
	torch_img = torch.from_numpy(arr).clone()
	torch_img_add_demension2 = torch.unsqueeze(torch_img,0)
	torch_img_add_demension2 = torch.unsqueeze(torch_img_add_demension2,0)

	return torch_img_add_demension2, original_width, original_height

# https://github.com/kazuto1011/grad-cam-pytorch
from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class _BaseWrapper(object):
	def __init__(self, model):
		super(_BaseWrapper, self).__init__()
		self.device = next(model.parameters()).device
		self.model = model
		self.handlers = []  # a set of hook function handlers

	def _encode_one_hot(self, ids):
		one_hot = torch.zeros_like(self.logits).to(self.device)
		one_hot.scatter_(1, ids, 1.0)
		return one_hot

	def forward(self, image):
		self.image_shape = image.shape[2:]
		self.logits = self.model(image)
		self.probs = F.softmax(self.logits, dim=1)
		return self.probs.sort(dim=1, descending=True)  # ordered results

	def backward(self, ids):
		"""
		Class-specific backpropagation
		"""
		one_hot = self._encode_one_hot(ids)
		self.model.zero_grad()
		self.logits.backward(gradient=one_hot, retain_graph=True)

	def generate(self):
		raise NotImplementedError

	def remove_hook(self):
		"""
		Remove all the forward/backward hook functions
		"""
		for handle in self.handlers:
			handle.remove()


class BackPropagation(_BaseWrapper):
	def forward(self, image):
		self.image = image.requires_grad_()
		return super(BackPropagation, self).forward(self.image)

	def generate(self):
		gradient = self.image.grad.clone()
		self.image.grad.zero_()
		return gradient

class GradCAM(_BaseWrapper):
	"""
	"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
	https://arxiv.org/pdf/1610.02391.pdf
	Look at Figure 2 on page 4
	"""

	def __init__(self, model, candidate_layers=None):
		super(GradCAM, self).__init__(model)
		self.fmap_pool = {}
		self.grad_pool = {}
		self.candidate_layers = candidate_layers  # list

		def save_fmaps(key):
			def forward_hook(module, input, output):
				self.fmap_pool[key] = output.detach()

			return forward_hook

		def save_grads(key):
			def backward_hook(module, grad_in, grad_out):
				self.grad_pool[key] = grad_out[0].detach()

			return backward_hook

		# If any candidates are not specified, the hook is registered to all the layers.
		for name, module in self.model.named_modules():
			if self.candidate_layers is None or name in self.candidate_layers:
				self.handlers.append(module.register_forward_hook(save_fmaps(name)))
				self.handlers.append(module.register_backward_hook(save_grads(name)))

	def _find(self, pool, target_layer):
		if target_layer in pool.keys():
			return pool[target_layer]
		else:
			raise ValueError("Invalid layer name: {}".format(target_layer))

	def generate(self, target_layer):
		fmaps = self._find(self.fmap_pool, target_layer)
		grads = self._find(self.grad_pool, target_layer)
		weights = F.adaptive_avg_pool2d(grads, 1)

		gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
		gcam = F.relu(gcam)
		gcam = F.interpolate(
			gcam, self.image_shape, mode="bilinear", align_corners=False
		)

		B, C, H, W = gcam.shape
		gcam = gcam.view(B, -1)
		gcam -= gcam.min(dim=1, keepdim=True)[0]
		gcam /= gcam.max(dim=1, keepdim=True)[0]
		gcam = gcam.view(B, C, H, W)

		return gcam	

import copy
import os.path as osp

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms


# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False


def get_device(cuda):
	cuda = cuda and torch.cuda.is_available()
	device = torch.device("cuda" if cuda else "cpu")
	if cuda:
		current_device = torch.cuda.current_device()
		print("Device:", torch.cuda.get_device_name(current_device))
	else:
		print("Device: CPU")
	return device


def load_images(image_paths):
	images = []
	raw_images = []
	
	image, raw_image = preprocess(image_paths)
	
	images.append(image)
	raw_images.append(raw_image)
	
	return images, raw_images


def get_classtable():
	classes = []
	with open("samples/synset_words.txt") as lines:
		for line in lines:
			line = line.strip().split(" ", 1)[1]
			line = line.split(", ", 1)[0].replace(" ", "_")
			classes.append(line)
	return classes


def get_enhance(img_name):
	# read image
	img = cv2.imread(img_name, cv2.IMREAD_COLOR)
	width = img.shape[1]
	height = img.shape[0]
	cropped_img = img[int(width*0.11): int(height*0.89), int(height*0.11): int(width*0.89)]
	# convert to LAB color space
	lab = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LAB)
	# separate channels
	L,A,B=cv2.split(lab)
	# compute minimum and maximum in 5x5 region using erode and dilate
	kernel = np.ones((5,5),np.uint8)
	min = cv2.erode(L,kernel,iterations = 1)
	max = cv2.dilate(L,kernel,iterations = 1)
	# convert min and max to floats
	min = min.astype(np.float64) 
	max = max.astype(np.float64) 
	# compute local contrast
	contrast = (max-min)/(max+min)
	np.nan_to_num(contrast, copy=False)
	# get average across whole image
	average_contrast = 100*np.mean(contrast)
	return average_contrast

def preprocess(image_path):
	raw_image = cv2.imread(image_path)
	#print(raw_image.shape)
	raw_image = cv2.resize(raw_image, (224,) * 2)

	####
	image1 = Image.open(image_path).convert("L")
	e = get_enhance(image_path)
	if e < 3.5:
	  img_enhance = ImageEnhance.Contrast(image1).enhance(1.35)   
	
	else:
	  img_enhance = image1

	width = image1.size[0]
	height = image1.size[1]
	croppedImage=img_enhance.crop((width*0.11, height*0.11, (width*0.89), (height*0.89)))                         # "R", "L" 마커 제거
	resize_img = croppedImage.resize((224, 224))
	re_img_arr=np.asarray(resize_img)
	arr = (re_img_arr - re_img_arr.min()) / (re_img_arr.max() - re_img_arr.min())
	tensor_img = torch.from_numpy(arr).clone()
	image = torch.unsqueeze(tensor_img, 0)

	####
	#print(raw_image.shape)
	#print(image.shape)
	return image, raw_image


# torchvision models
model_names = sorted(
	name
	for name in models.__dict__
	if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def demo1(net0, image_paths, target_layer, arch, topk, output_dir):
	"""
	Visualize model responses given multiple images
	"""

	device = torch.device('cpu')

	# Synset words
	classes = [0, 1, 2, 3]

	# Model from torchvision
	#model = models.__dict__[arch](pretrained=True)
	model = net0
	model.to(device)
	#model.eval()

	# Images
	images, raw_images = load_images(image_paths)
	dtype = torch.float
	images = torch.stack(images).to(device=device, dtype = torch.float)

	"""
	Common usage:
	1. Wrap your model with visualization classes defined in grad_cam.py
	2. Run forward() with images
	3. Run backward() with a list of specific classes
	4. Run generate() to export results
	"""

	bp = BackPropagation(model=model)
	probs, ids = bp.forward(images)  # sorted
	#print("demo ids: ",ids)
	print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

	gcam = GradCAM(model=model)
	_ = gcam.forward(images)

	gcam_list = []
	
	for i in range(3):
		# Grad-CAM
		gcam.backward(ids=ids[:, [i]])
		regions = gcam.generate(target_layer=target_layer)

		for j in range(len(images)):
			print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

		gcam_list.append(regions[0,0])
	return gcam_list, probs, ids


class BBoxerwGradCAM():
	
	def __init__(self,learner,heatmaps,probs,ids,image_path, file_path, resize_scale_list,bbox_scale_list):
		self.learner = learner
		self.heatmaps = heatmaps
		self.probs = probs.detach().numpy()
		self.ids = ids.numpy()
		self.image_path = image_path
		self.file_path = file_path
		self.resize_list = resize_scale_list
		self.scale_list = bbox_scale_list
		
		self.og_img, self.smooth_heatmaps = self.heatmap_smoothing()
		
		self.bbox_coords1, self.grey_img_list, self.contours_list = self.form_largest_bboxes()     # 원래코드 :  self.form_bboxes()
		
	def heatmap_smoothing(self):
		og_img = cv2.imread(self.image_path)
		og_img = cv2.resize(og_img, (self.resize_list[0],self.resize_list[1])) # Resizing

		heatmapshows = []
		for heatmap in self.heatmaps: #np.float32(gcam.cpu())
		  heatmap = np.float32(heatmap.cpu())
		  heatmap = cv2.resize(heatmap, (self.resize_list[0],self.resize_list[1])) # Resizing
		  '''
		  The minimum pixel value will be mapped to the minimum output value (alpha - 0)
		  The maximum pixel value will be mapped to the maximum output value (beta - 155)
		  Linear scaling is applied to everything in between.
		  These values were chosen with trial and error using COLORMAP_JET to deliver the best pixel saturation for forming contours.
		  '''
		  heatmapshow = cv2.normalize(heatmap, None, alpha=-150, beta=125, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)    # heatmap rgb값의 최소값과 최대값을 조정, beta 값이 작아질수록 heatmap의 가장 붉은부분부분만 뽑아냄(110~155 사이에서 조정하길 추천), alpha 값을 마이너스로 둬서 대비(?)가 더 커지게 함
		  heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
		  heatmapshows.append(heatmapshow)

		return og_img, heatmapshows
	
	def show_bboxrectangle(self):
		# 가장 큰 bbox
		classes=[]
		for i in self.ids[0]:
			if i == 0:
				classes.append("normal")
			elif i==1:
				classes.append("covid19")
			elif i==2:
				classes.append("bacterial")
			elif i==3:
				classes.append("viral")
		print("show box rectangle")

		length0 = self.bbox_coords1[0][0]+self.bbox_coords1[0][1]+self.bbox_coords1[0][2]+self.bbox_coords1[0][3]
		length1 = self.bbox_coords1[1][0]+self.bbox_coords1[1][1]+self.bbox_coords1[1][2]+self.bbox_coords1[1][3]
		length2 = self.bbox_coords1[2][0]+self.bbox_coords1[2][1]+self.bbox_coords1[2][2]+self.bbox_coords1[2][3]
		
		print("len0: ", length0)
		print("self.prob: ",self.probs[0][0])
		if length0 > 0:
			
			if self.probs[0][0] > 0.01:
				cv2.rectangle(self.og_img,      # 이미지 파일
							(self.bbox_coords1[0][0],self.bbox_coords1[0][1]),       # 시작점 좌표
							(self.bbox_coords1[0][0]+self.bbox_coords1[0][2],self.bbox_coords1[0][1]+self.bbox_coords1[0][3]),      # 종료점 좌표
							(0,0,255),2)      # 직사각형 색상, 선 두께

				if self.bbox_coords1[0][1] <=30:
					cv2.putText(self.og_img, str(classes[0]), (self.bbox_coords1[0][0],self.bbox_coords1[0][1]+self.bbox_coords1[0][3]+14),
						  cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
					cv2.putText(self.og_img, str(format(self.probs[0][0],"<10.2%")), (self.bbox_coords1[0][0],self.bbox_coords1[0][1]+self.bbox_coords1[0][3]+28),
						  cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
				elif ((self.bbox_coords1[0][0]+self.bbox_coords1[0][2])>=(self.og_img.shape[1]-30)):
					cv2.putText(self.og_img, str(classes[0]), (self.bbox_coords1[0][0]+self.bbox_coords1[0][2]-80,self.bbox_coords1[0][1]-21),
						  cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
					cv2.putText(self.og_img, str(format(self.probs[0][0],"<10.2%")), (self.bbox_coords1[0][0]+self.bbox_coords1[0][2]-80,self.bbox_coords1[0][1]-7),
						  cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
				elif (self.bbox_coords1[0][1] <=30 and ((self.bbox_coords1[0][0]+self.bbox_coords1[0][2])>=(self.og_img.shape[1]-30))):
					cv2.putText(self.og_img, str(classes[0]), (self.bbox_coords1[0][0]+self.bbox_coords1[0][2]-80,bbox_coords1[0][1]+self.bbox_coords1[0][3]+14),
						  cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
					cv2.putText(self.og_img, str(format(self.probs[0][0],"<10.2%")), (self.bbox_coords1[0][0]+self.bbox_coords1[0][2]-80,self.bbox_coords1[0][1]+self.bbox_coords1[0][3]+28),
						  cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
				else:
					cv2.putText(self.og_img, str(classes[0]), (self.bbox_coords1[0][0],self.bbox_coords1[0][1]-21),
						  cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
					cv2.putText(self.og_img, str(format(self.probs[0][0],"<10.2%")), (self.bbox_coords1[0][0],self.bbox_coords1[0][1]-7),
						  cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
		
		if length1 > 0:
			if self.probs[0][1] > 0.01:
				cv2.rectangle(self.og_img,      # 이미지 파일
						(self.bbox_coords1[1][0],self.bbox_coords1[1][1]),       # 시작점 좌표
						(self.bbox_coords1[1][0]+self.bbox_coords1[1][2],self.bbox_coords1[1][1]+self.bbox_coords1[1][3]),      # 종료점 좌표
						(255,0,0),2)      # 직사각형 색상, 선 두께

				if self.bbox_coords1[1][1] <=30:
					cv2.putText(self.og_img, str(classes[1]), (self.bbox_coords1[1][0],self.bbox_coords1[1][1]+self.bbox_coords1[1][3]+14),
						  cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
					cv2.putText(self.og_img, str(format(self.probs[0][1],"<10.2%")), (self.bbox_coords1[1][0],self.bbox_coords1[1][1]+self.bbox_coords1[1][3]+28),
						  cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
				elif ((self.bbox_coords1[1][0]+self.bbox_coords1[1][2])>=(self.og_img.shape[1]-30)):
					cv2.putText(self.og_img, str(classes[1]), (self.bbox_coords1[1][0]+self.bbox_coords1[1][2]-80,self.bbox_coords1[1][1]-21),
						  cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
					cv2.putText(self.og_img, str(format(self.probs[0][1],"<10.2%")), (self.bbox_coords1[1][0]+self.bbox_coords1[1][2]-80,self.bbox_coords1[1][1]-7),
						  cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
				elif (self.bbox_coords1[1][1] <=30 and ((self.bbox_coords1[1][0]+self.bbox_coords1[1][2])>=(self.og_img.shape[1]-30))):
					cv2.putText(self.og_img, str(classes[1]), (self.bbox_coords1[1][0]+self.bbox_coords1[1][2]-80,bbox_coords1[1][1]+self.bbox_coords1[1][3]+14),
						  cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
					cv2.putText(self.og_img, str(format(self.probs[0][1],"<10.2%")), (self.bbox_coords1[1][0]+self.bbox_coords1[1][2]-80,self.bbox_coords1[1][1]+self.bbox_coords1[1][3]+28),
						  cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
				else:
					cv2.putText(self.og_img, str(classes[1]), (self.bbox_coords1[1][0],self.bbox_coords1[1][1]-21),
						  cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
					cv2.putText(self.og_img, str(format(self.probs[0][1],"<10.2%")), (self.bbox_coords1[1][0],self.bbox_coords1[1][1]-7),
						  cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
		
		if length2>0:
			if self.probs[0][2] > 0.01:
				cv2.rectangle(self.og_img,      # 이미지 파일
							(self.bbox_coords1[2][0],self.bbox_coords1[2][1]),       # 시작점 좌표
							(self.bbox_coords1[2][0]+self.bbox_coords1[2][2],self.bbox_coords1[2][1]+self.bbox_coords1[2][3]),      # 종료점 좌표
							(51,102,51),2)      # 직사각형 색상, 선 두께

				if self.bbox_coords1[2][1] <=30:
					cv2.putText(self.og_img, str(classes[2]), (self.bbox_coords1[2][0],self.bbox_coords1[2][1]+self.bbox_coords1[2][3]+14),
						  cv2.FONT_HERSHEY_PLAIN, 1, (51,102,51), 1)
					cv2.putText(self.og_img, str(format(self.probs[0][2],"<10.2%")), (self.bbox_coords1[2][0],self.bbox_coords1[2][1]+self.bbox_coords1[2][3]+28),
						  cv2.FONT_HERSHEY_PLAIN, 1, (51,102,51), 1)
				elif ((self.bbox_coords1[2][0]+self.bbox_coords1[2][2])>=(self.og_img.shape[1]-30)):
					cv2.putText(self.og_img, str(classes[2]), (self.bbox_coords1[2][0]+self.bbox_coords1[2][2]-80,self.bbox_coords1[2][1]-21),
						  cv2.FONT_HERSHEY_PLAIN, 1, (51,102,51), 1)
					cv2.putText(self.og_img, str(format(self.probs[0][2],"<10.2%")), (self.bbox_coords1[2][0]+self.bbox_coords1[2][2]-80,self.bbox_coords1[2][1]-7),
						  cv2.FONT_HERSHEY_PLAIN, 1, (51,102,51), 1)
				elif (self.bbox_coords1[2][1] <=30 and ((self.bbox_coords1[2][0]+self.bbox_coords1[2][2])>=(self.og_img.shape[1]-30))):
					cv2.putText(self.og_img, str(classes[2]), (self.bbox_coords1[2][0]+self.bbox_coords1[2][2]-80,bbox_coords1[2][1]+self.bbox_coords1[2][3]+14),
						  cv2.FONT_HERSHEY_PLAIN, 1, (51,102,51), 1)
					cv2.putText(self.og_img, str(format(self.probs[0][2],"<10.2%")), (self.bbox_coords1[2][0]+self.bbox_coords1[2][2]-80,self.bbox_coords1[2][1]+self.bbox_coords1[3][3]+28),
						  cv2.FONT_HERSHEY_PLAIN, 1, (51,102,51), 1)
				else:
					cv2.putText(self.og_img, str(classes[2]), (self.bbox_coords1[2][0],self.bbox_coords1[2][1]-21),
						  cv2.FONT_HERSHEY_PLAIN, 1, (51,102,51), 1)
					cv2.putText(self.og_img, str(format(self.probs[0][2],"<10.2%")), (self.bbox_coords1[2][0],self.bbox_coords1[2][1]-7),
						  cv2.FONT_HERSHEY_PLAIN, 1, (51,102,51), 1)       
		   
		cv2.imwrite(self.file_path, np.uint8(self.og_img))

	# 가장 큰 bounding box 형성하는 부분(원래 코드에서 개선한 코드)
	def form_largest_bboxes(self):
		
		coordinate_list = []
		grey_img_list = []
		contours_list = []
		for idx, smooth_heatmap in enumerate(self.smooth_heatmaps):
			try:
				print(idx)
				grey_img = cv2.cvtColor(smooth_heatmap, cv2.COLOR_BGR2GRAY)
				ret,thresh = cv2.threshold(grey_img,150,255,cv2.THRESH_BINARY)      # threshold함수 기준값 변경함으로써 등고선 조정
				contours,hierarchy = cv2.findContours(thresh, 1, 2)

				contour_areas = []   # contour 면적 계산
				for i, c in enumerate(contours):
					contour_areas.append(cv2.contourArea(c))

				sorted_contours = sorted(zip(contour_areas, contours), key=lambda x:x[0], reverse=True)    # contour 면적이 큰 순서대로 정렬

				biggest_contour= sorted_contours[0][1]     # contour 면적이 큰 contours 선택
				x1,y1,w1,h1 = cv2.boundingRect(biggest_contour)
				x1 = int(x1*self.scale_list[0]) # rescaling the boundary box based on user input
				y1 = int(y1*self.scale_list[1])
				w1 = int(w1*self.scale_list[2])
				h1 = int(h1*self.scale_list[3])

				coordinate_list.append([x1,y1,w1,h1])
				grey_img_list.append(grey_img)
				contours_list.append(contours)
			except:
				coordinate_list.append([0,0,0,0])
				grey_img_list.append(grey_img)
				contours_list.append(contours)

		return coordinate_list, grey_img_list, contours_list


def main(target_img_path, filename):
	# image 파일명
	target_img_path = target_img_path.split('/')[-1]
	# image 경로
	target_image_path = 'pyflask/static/pyimages/'+ target_img_path          # 타깃 이미지  

	# 이미지 전처리 
	target_torch_image, origin_wid, origin_hei = preprocess_image(target_image_path) # creates img to a constant tensor
 
	# 학습 모델 불러오기
	load_model = GoogLeNet.GoogLeNet()
	path = "pyflask/model/"
	load_model.load_state_dict(torch.load(path+"googlenet_cad_addv_new_crop_enhance35_model_state_dict_epoch25.pt", map_location='cpu'),strict=False)
	load_model.eval()
   
	# 바이너리 모델 불러오기
	load_binary_model = GoogLeNetBinary.GoogLeNetBinary()
	load_binary_model.load_state_dict(torch.load(path+'googlenet_binary_model_state_dict_epoch25.pt', map_location='cpu'),strict=False)
	load_binary_model.eval()
   
	# 바이너리 예측하기
	pred_list_binary = []

	device = torch.device('cpu')
	dtype = torch.float
   
	x = target_torch_image.to(device=device, dtype=dtype)  

	scores = load_binary_model(x)
	_, preds = scores.max(dim=1)
	pred_list_binary += preds.to("cpu")

	pred_class_binary = pred_list_binary[0].numpy()
	print("g_p_t:", pred_class_binary)

	# 예측하기
	pred_list = []
 
	device = torch.device('cpu')
	dtype = torch.float
   
	x = target_torch_image.to(device=device, dtype=dtype)  

	scores = load_model(x)
	_, preds = scores.max(dim=1)
	pred_list += preds.to("cpu")

	pred_class = pred_list[0].numpy()
   
	prediction_var = Variable(x, requires_grad=True)
	prediction = load_model(prediction_var.to(device=device, dtype=dtype))
	pred_probabilities = F.softmax(prediction, dim = 1).data.squeeze()
   
	np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.4f}".format(x)})
	class_idx_list = topk(pred_probabilities,4)[0].float().numpy()
	class_idx_list = np.round_(class_idx_list*100,3)
   
	# 그래드캠 얻기
	grad_model = load_model
   
	gcam, probs, ids = demo1(grad_model, target_image_path, "inception5b", "googlenet", 4, "pyflask/static/pyimages/grad_img")
   

	# 박스 그리기
	filepath = 'pyflask/static/pyimages/grad_img/'+filename
   
	returnpath = './pyimages/grad_img/'+filename
   
	if origin_wid>300:
		rate = 300/origin_wid
		hei=int(origin_hei*rate)
		image_resizing_scale = [300,hei]
	else:
		image_resizing_scale = [origin_wid,origin_hei]
		
	bbox_scaling = [1,1,1,1] 

	bbox = BBoxerwGradCAM(grad_model,
						  gcam,
						  probs,
						  ids,
						  target_image_path,
						  filepath,
						  image_resizing_scale,
						  bbox_scaling)
	
	bbox.show_bboxrectangle()
   
	return pred_class, returnpath, class_idx_list, pred_class_binary
if __name__ == "__main__":
	main()
