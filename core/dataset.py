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



import time
cv2.ocl.setUseOpenCL(False)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),                  
        transforms.RandomHorizontalFlip(p=0.5),         
        transforms.RandomRotation(degrees=10),         
        transforms.RandomAutocontrast(p=0.3),          
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        transforms.ToTensor(),                         
        #transforms.Normalize(mean = [0.54, 0.39, 0.39],
        #                     std = [0.25, 0.22, 0.22]), 
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std  = [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),  
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),                  
        transforms.ToTensor(),                          
        #transforms.Normalize(mean = [0.54, 0.39, 0.39],
        #                     std = [0.25, 0.22, 0.22]),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std  = [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),                  
        transforms.ToTensor(),                          
        #transforms.Normalize(mean = [0.54, 0.39, 0.39],
        #                     std = [0.25, 0.22, 0.22]),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std  = [0.229, 0.224, 0.225]),
    ])
}

# =========================
#  完全随机帧采样
# =========================
def sample_random_frames(total_frames, num_samples=5):
    """ 在视频中完全随机抽取 num_samples 帧 """
    if total_frames < num_samples:
        return list(range(total_frames))  # 如果视频帧数小于所需帧数，则返回所有帧
    
    indices = sorted(random.sample(range(total_frames), num_samples))  # 随机选取 num_samples 帧
    return indices

# =========================
#  人脸检测
# =========================
model = YOLO("yolov5s.pt")  

def detect_face(img, last_valid_face=None):
    """
    使用 YOLO 检测人脸，优先选择置信度最高的一个。如果没有检测到，则回退使用 last_valid_face。
    """
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



# =========================
#  数据集
# =========================
class MMDataset(Dataset):
    def __init__(self,dataset='mosi', mode='train', image_num = 5, generate_num = 1):
        self.dataset=dataset
        self.mode = mode
        self.dataPath = '/scratch/song.xinwe/'+self.dataset+'/unaligned_50.pkl'
        self.train_mode = 'regression'
        self.image_num = image_num
        self.transform = data_transforms[self.mode]
        self.seq_lens = [50,50]
        self.generate_num = generate_num
        self.__init_dataset()
    
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
        self.video_paths = ['/scratch/song.xinwe/'+self.dataset+'/Raw/'+('/').join(video_id.split('$_$'))+'.mp4' for video_id in data[self.mode]['id']]
        self.cleaned_images = []
        
        #self.__truncated()
            

        for text, audio, label, video_path in tqdm(zip(self.text, self.audio, self.labels, self.video_paths),total = len(self.text),position=0 
                                                   ,leave=True):
            for _ in range(self.generate_num):
                images = self.__load_images(video_path)
                if images is not None and len(images) > 0:
                    self.cleaned_text.append(text)
                    self.cleaned_audio.append(audio)
                    self.cleaned_labels.append(label)
                    self.cleaned_images.append(images)

 
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
        max_search_window = 3  # 最大偏移距离
        last_valid_face = None

        def find_nearby_face(idx):
            """从邻域帧中寻找能检测到人脸的帧"""
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
            return None  # 所有邻居都失败

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

        while len(images) < self.image_num:
            images.append(last_valid_face if last_valid_face is not None else np.zeros((224, 224, 3), dtype=np.uint8))

        return images


        
    def __truncated(self):
        def Truncated(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+length])
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+length])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
                       
        audio_length, video_length = self.seq_lens 
        self.audio = Truncated(self.audio, audio_length)

    def __len__(self):
        return len(self.cleaned_labels)

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
