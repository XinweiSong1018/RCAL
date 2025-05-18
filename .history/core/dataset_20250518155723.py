'''
* @name: yolov5s.pt
* @description: Pre-trained YOLOv5 small model for object detection.
* The model is used here specifically for face detection in video frames.
* Source: https://github.com/ultralytics/yolov5
'''

# Import essential libraries for data handling, model loading, image processing, and video reading
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

import pickle
import numpy as np
import torch
import cv2
import random
from torchvision import models
from decord import VideoReader, cpu
from tqdm import tqdm
from ultralytics import YOLO



# Disable OpenCL to avoid potential conflicts with some OpenCV operations
import time
cv2.ocl.setUseOpenCL(False)

# Define image transformations for different dataset splits (train, valid, test)
# Includes standard resizing, normalization, and data augmentation for training
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),                  
        transforms.RandomHorizontalFlip(p=0.5),         
        transforms.RandomRotation(degrees=10),         
        transforms.RandomAutocontrast(p=0.3),          
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        transforms.ToTensor(),                         
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std  = [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),  
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),                  
        transforms.ToTensor(),                          
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std  = [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),                  
        transforms.ToTensor(),                          
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std  = [0.229, 0.224, 0.225]),
    ])
}

# Randomly sample frame indices from a video
def sample_random_frames(total_frames, num_samples=5):
    if total_frames < num_samples:
        return list(range(total_frames))  
    
    indices = sorted(random.sample(range(total_frames), num_samples))  
    return indices

# Load YOLOv5s for face detection from frames
model = YOLO("yolov5s.pt")  

# Detect face in an image using YOLOv5. Returns cropped face or last valid face if detection fails.
def detect_face(img, last_valid_face=None):
    
    results = model.predict(source=img, device=0 if torch.cuda.is_available() else 'cpu', verbose=False)

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return last_valid_face

    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    best_idx = np.argmax(scores)
    x1, y1, x2, y2 = map(int, boxes[best_idx])

    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face = img[y1:y2, x1:x2]

    return face if face.size > 0 else last_valid_face


# Custom Dataset class for loading multimodal (text, audio, image) data
class MMDataset(Dataset):
    def __init__(self,args, mode='train'):
        self.dataset=args.dataset.datasetName
        self.mode = mode
        if self.dataset=='chsims':
            self.dataPath = 'data/'+self.dataset+'/unaligned_39.pkl'
        else:
            self.dataPath = 'data/'+self.dataset+'/unaligned_50.pkl'
        self.train_mode = args.dataset.train_mode
        self.image_num = args.dataset.image_num
        self.transform = data_transforms[self.mode]
        if mode=='train':
            self.generate_num = 4
        else:
            self.generate_num = 1# data augmentation
        self.__init_dataset()
    # Initialize the dataset by loading pickled features and processing valid samples
    def __init_dataset(self):
        with open(self.dataPath, 'rb') as f:
            data = pickle.load(f)


        self.text = data[self.mode]['text_bert'].astype(np.float32)
        self.cleaned_text = []
     
        
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.cleaned_audio = []

        
        self.labels = data[self.mode][self.train_mode+'_labels'].astype(np.float32)
        self.cleaned_labels = []
        
        self.ids = data[self.mode]['id']
        self.video_paths = ['data/'+self.dataset+'/Raw/'+('/').join(video_id.split('$_$'))+'.mp4' for video_id in data[self.mode]['id']]
        self.cleaned_images = []
        
       

        for text, audio, label, video_path in tqdm(zip(self.text, self.audio, self.labels, self.video_paths),total = len(self.text),position=0 
                                                   ,leave=True):
            for _ in range(self.generate_num):
                images = self.__load_images(video_path)
                if images is not None and len(images) > 0:
                    self.cleaned_text.append(text)
                    self.cleaned_audio.append(audio)
                    self.cleaned_labels.append(label)
                    self.cleaned_images.append(images)

     # Extract faces from selected frames in a video
    def __load_images(self, video_path):
        try:
            vr = VideoReader(video_path, ctx=cpu())
            total_frames = len(vr)
        except Exception as e:
            print(f"[ERROR] Cannot open video: {video_path} — {e}")
            return None

        if total_frames < self.image_num:
            return None

        selected_frames = sorted(random.sample(range(total_frames), self.image_num))
        images = []
        max_search_window = 3 
        last_valid_face = None
  # Try to find nearby frame with valid face if current one fails
        def find_nearby_face(idx):
            offsets = [0] + [i for j in range(1, max_search_window + 1) for i in (j, -j)]  # e.g., [0, 1, -1, 2, -2, ...]
            for offset in offsets:
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < total_frames:
                    try:
                        frame = vr[neighbor_idx].asnumpy().astype(np.uint8)
                        face = detect_face(frame, None)
                        if face is not None:
                            return face
                    except Exception as e:
                        continue
            return None  
        # Process selected frames
        for idx in selected_frames:
            try:
                frame = vr[idx].asnumpy().astype(np.uint8)
                face_image = detect_face(frame, None)

                if face_image is not None:
                    images.append(face_image)
                    last_valid_face = face_image
                else:
                    nearby_face = find_nearby_face(idx)
                    if nearby_face is not None:
                        images.append(nearby_face)
                        last_valid_face = nearby_face
                    else:
                        images.append(last_valid_face if last_valid_face is not None else frame)
            except Exception as e:
                print(f"[WARNING] Failed to decode frame {idx} from {video_path}: {e}")
                images.append(last_valid_face if last_valid_face is not None else np.zeros((224, 224, 3), dtype=np.uint8))
        # Pad frames if not enough were found
        while len(images) < self.image_num:
            images.append(last_valid_face if last_valid_face is not None else np.zeros((224, 224, 3), dtype=np.uint8))

        return images


    # Return total number of usable samples
    def __len__(self):
        return len(self.cleaned_labels)
    # Return a single multimodal sample: text, audio, image sequence, label
    def __getitem__(self, index):
        faces = [Image.fromarray(image) if isinstance(image,np.ndarray) else image for image in self.cleaned_images[index]]
        faces = [self.transform(image) for image in faces]
        sample = {
            'text': torch.Tensor(self.cleaned_text[index]), 
            'audio': torch.Tensor(self.cleaned_audio[index]),
            'images': torch.stack(faces,dim=0),
            'labels': torch.tensor(self.cleaned_labels[index], dtype=torch.float32),
            'index': index
        } 
        return sample
