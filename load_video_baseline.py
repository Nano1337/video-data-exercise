import av
import time
from more_itertools import one
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
from tqdm import tqdm


class VideoDataset(Dataset):
    def __init__(self, filename, transform, use_gpu=False):
        self.filename = filename
        self.transform = transform
        self.use_gpu = use_gpu
        self.container = None
        self.stream = None

        # get frame count
        container = av.open(filename) # container handles all components of the video file (audio, video, metadata, etc.)
        stream = one(container.streams.video) # ensures only one video stream is retrieved
        self.num_frames = stream.frames
        # close this and have each worker re-open handle to container in its own process
        container.close()
        
    def __len__(self):
        return self.num_frames  # num frames in the video

    def __getitem__(self, idx):
        image = self.read_frame_gpu(idx) if self.use_gpu else self.read_frame(idx)
        image = self.transform(image)
        return image

    def init_container(self):
        self.container = av.open(self.filename)
        self.stream = one(self.container.streams.video)

    def read_frame(self, idx):
        if self.container is None:
            self.init_container()

        # finds the frame with the given pts (presentation timestamp)
        target_pts = idx * int(self.stream.duration / self.stream.frames)
        self.container.seek(target_pts, backward=True, any_frame=False, stream=self.stream)

        # returns the frame as a PIL image
        for frame in self.container.decode(video=0):
            if frame.pts == target_pts:
                return frame.to_image()
        raise ValueError(f'Could not find frame with pts {target_pts}')

    def read_frame_gpu(self, idx):
        if self.container is None:
            self.init_container()

        # Create hardware-accelerated decoder. Note that attempting to re-use this between 
        # calls to read_frame_gpu breaks: the codec appears to be stateful!
        # this means that frames have to be decoded sequentially somehow due to its reliance on neighboring frames
        self.ctx = av.Codec('h264_cuvid', 'r').create()
        self.ctx.extradata = self.stream.codec_context.extradata

        # finds the frame with the given pts (presentation timestamp)
        target_pts = idx * int(self.stream.duration / self.stream.frames)
        self.container.seek(target_pts, backward=True, stream=self.stream, any_frame=False)

        # returns the frame as a PIL image
        for packet in self.container.demux(self.stream):
            for frame in self.ctx.decode(packet):
                if frame.pts == target_pts:
                    return frame.to_image()
        raise ValueError(f'Could not find frame with pts {target_pts}')


def run_dataloader():
    filename = 'IMG_4475.MOV'
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
        transforms.ToTensor(),  # CHW, float
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    ])

    ds = VideoDataset(filename, transform=train_transform, use_gpu=False) # CHANGE HERE for GPU support 
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