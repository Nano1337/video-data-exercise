import av
import os
import random
from more_itertools import one
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
from tqdm import tqdm
import time
from collections import defaultdict

class MultiVideoDataset(Dataset):
    """
    Method inspired by https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch/blob/main/video_dataset.py
    """

    def __init__(self, filenames, transform, num_segments=3, frames_per_segment=1, use_gpu=False):
        self.filenames = filenames
        self.transform = transform
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.use_gpu = use_gpu
        self.segment_metadata = []
        self.total_segments = 0

        for filename in filenames:
            container = av.open(filename)
            stream = one(container.streams.video)
            num_frames = stream.frames
            segment_length = num_frames // num_segments
            for segment_idx in range(num_segments):
                start_frame = segment_idx * segment_length
                end_frame = start_frame + segment_length
                self.segment_metadata.append((filename, start_frame, end_frame))
            self.total_segments += num_segments
            container.close()

    def __len__(self):
        return self.total_segments * self.frames_per_segment

    def __getitem__(self, _):
        segment_idx = random.randint(0, self.total_segments - 1)
        filename, start_frame, end_frame = self.segment_metadata[segment_idx]
        frame_idx = random.randint(start_frame, end_frame - 1)

        image = self.read_frame(frame_idx, filename)
        image = self.transform(image)
        return image

    def read_frame(self, idx, filename):
        try:
            container = av.open(filename)
            stream = one(container.streams.video)

            target_pts = idx * int(stream.duration / stream.frames)
            container.seek(target_pts, backward=True, any_frame=False, stream=stream)

            for frame in container.decode(video=0):
                if frame.pts == target_pts:
                    return frame.to_image()
        except Exception as e:
            print(f"Error reading frame: {e}")
        finally:
            container.close()
        return None

def run_dataloader():
    foldername = "video_data"
    filelist = os.listdir(foldername)
    filelist = [os.path.join(foldername, f) for f in filelist]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    ])

    ds = MultiVideoDataset(filelist, train_transform, num_segments=1, frames_per_segment=5, use_gpu=False)
    train_dataloader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=10)
    device = 'cuda'

    run_model = False

    if run_model:
        net = nn.Conv2d(3, 64, 3, 2)
        net.to(device)

    it = 0
    start = time.time()
    for imgs in tqdm(train_dataloader):
        print(imgs.shape)
        imgs = imgs.to(device)
        if run_model:
            with torch.no_grad():
                net(imgs)
        it += 1

    print(f'it/sec: {it/(time.time() - start)}')

if __name__ == "__main__":
    run_dataloader()
