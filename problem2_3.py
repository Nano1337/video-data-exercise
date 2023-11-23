import av
import os
import random
import time
from more_itertools import one
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
from tqdm import tqdm

class MultiVideoDataset(Dataset):
    def __init__(self, filenames, transform, use_gpu=False):
        self.filenames = filenames
        self.transform = transform
        self.use_gpu = use_gpu
        self.global_frame_indices = []

        for filename in filenames:
            container = av.open(filename)
            stream = one(container.streams.video)
            num_frames = stream.frames
            container.close()
            for frame_number in range(num_frames):
                self.global_frame_indices.append((filename, frame_number))

    def __len__(self):
        return len(self.global_frame_indices)

    def __getitem__(self, idx):
        filename, frame_number = self.global_frame_indices[idx]
        image = self.read_frame(frame_number, filename)
        image = self.transform(image)
        return image

    def read_frame(self, frame_number, filename):
        with av.open(filename) as container:
            stream = one(container.streams.video)
            target_pts = frame_number * int(stream.duration / stream.frames)
            container.seek(target_pts, backward=True, any_frame=False, stream=stream)
            for frame in container.decode(video=0):
                if frame.pts == target_pts:
                    return frame.to_image()
        raise ValueError(f'Could not find frame with pts {target_pts}')

def run_dataloader():
    foldername = "video_data"
    filelist = os.listdir(foldername)
    filelist = [os.path.join(foldername, f) for f in filelist]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    ])

    ds = MultiVideoDataset(filelist, train_transform, use_gpu=False)
    train_dataloader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=10)
    device = 'cuda'

    run_model = False

    if run_model:
        net = nn.Conv2d(3, 64, 3, 2)
        net.to(device)

    it = 0
    start = time.time()
    for imgs in tqdm(train_dataloader):
        imgs = imgs.to(device)
        if run_model:
            with torch.no_grad():
                net(imgs)
        it += 1

    print(f'it/sec: {it/(time.time() - start)}')

if __name__ == "__main__":
    run_dataloader()
