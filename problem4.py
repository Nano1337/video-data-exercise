import av
import os
from more_itertools import one
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
from tqdm import tqdm
import time
from einops import rearrange

class MultiVideoDataset(Dataset):
    def __init__(self, filenames, transform, chunk_size=10, use_gpu=False):
        self.filenames = filenames
        self.transform = transform
        self.chunk_size = chunk_size
        self.use_gpu = use_gpu
        self.global_chunk_indices = []

        for filename in filenames:
            container = av.open(filename)
            stream = one(container.streams.video)
            num_frames = stream.frames
            container.close()
            
            # Calculate the number of chunks in each video
            num_chunks = num_frames // self.chunk_size
            for chunk_start in range(num_chunks):
                self.global_chunk_indices.append((filename, chunk_start * self.chunk_size))

    def __len__(self):
        return len(self.global_chunk_indices)

    def __getitem__(self, idx):
        filename, start_frame = self.global_chunk_indices[idx]
        end_frame = start_frame + self.chunk_size
        frames = self.read_frames(start_frame, end_frame, filename)
        images = [self.transform(frame) for frame in frames]
        return torch.stack(images)  # Stack images into a single tensor

    def read_frames(self, start_frame, end_frame, filename):
        frames = []
        with av.open(filename) as container:
            stream = one(container.streams.video)
            target_start_pts = start_frame * int(stream.duration / stream.frames)

            container.seek(target_start_pts, backward=True, any_frame=False, stream=stream)
            for frame in container.decode(video=0):
                if frame.pts >= target_start_pts:
                    frames.append(frame.to_image())
                    if len(frames) >= (end_frame - start_frame):
                        break

        if len(frames) < (end_frame - start_frame):
            raise ValueError(f'Could not read all frames from {start_frame} to {end_frame}')

        return frames

def custom_collate_fn(batch):
    batch = torch.stack(batch, dim=0)
    batch = rearrange(batch, 'b f c h w -> (b f) c h w')
    print(batch.shape)
    return batch[:128]  # Ensure the batch size is 128 frames


def run_dataloader():
    foldername = "video_data"
    filelist = os.listdir(foldername)
    filelist = [os.path.join(foldername, f) for f in filelist]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    ])

    ds = MultiVideoDataset(filelist, train_transform, chunk_size=8, use_gpu=False)
    train_dataloader = DataLoader(
        ds, 
        batch_size=128, 
        shuffle=True, 
        num_workers=10,
        collate_fn=custom_collate_fn,
        )
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
