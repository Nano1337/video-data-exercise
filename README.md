# Video Dataloader Exercise

Author: Haoli Yin

# Installation and Setup

This starter codebase consists of a PyTorch Dataloader for videos, which reads frames at uniform random from a .MOV file, preprocesses them with some transforms, and then batches this together in parallel into a batch size of 128. 

1. Create python virtual environment

```
python -m venv ./venv
source venv/bin/activate
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Download the video to use: 
```
gdown 1PLCIa9lvlKMEJjSahixoChDsF_s7xFuL
```

4. Run the following script with use_cuda=False

```
python load_video_baseline.py
```

## Hardware-Accelerated FFmpeg:
Building FFmpeg from source for linux-based systems and ensuring that the FFmpeg library path is added to your system's dynamic linker configuration is a multi-step process. 

Some commands were taken from the (following link)[https://www.cyberciti.biz/faq/how-to-install-ffmpeg-with-nvidia-gpu-acceleration-on-linux] on how to install ffmpeg with nvidia gpu acceleration on linux but a lot of it was by trial and error.

Below is a step-by-step guide:

### Step 1: Install Necessary Dependencies

Before building FFmpeg, you need to install various dependencies required for the compilation process.

1. Open a terminal.
2. Update your package manager's database (example for Debian/Ubuntu):
   ```bash
   sudo apt update
   ```
3. Install the build dependencies (example for Debian/Ubuntu):
   ```bash
   sudo apt install autoconf automake build-essential cmake git-core libass-dev libfreetype6-dev \
   libsdl2-dev libtool libva-dev libvdpau-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev \
   pkg-config texinfo wget zlib1g-dev yasm libx264-dev libx265-dev libnuma-dev \
   libvpx-dev libfdk-aac-dev libmp3lame-dev libopus-dev
   ```

   This is a general list and might need adjustments based on your specific requirements for FFmpeg features.

### Step 2: Download nv-codec-headers

1. To compile ffmpeg from NVIDIA, we need ffnvcodec. Clone git repo: 
    ```bash 
    mkdir ~/nvidia/ && cd ~/nvidia/
    git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
    ```
2. Now build it: 
    ```bash
    cd nv-codec-headers && sudo make install
    ```

### Step 3: Download FFmpeg Source Code

1. Clone the FFmpeg source code repository:
   ```bash
   cd ~/nvidia
   git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
   ```
2. Navigate to the cloned directory:
   ```bash
   cd ffmpeg
   ```

### Step 3: Configure the Build

Run the `configure` script to set up the build environment. Here's an example command that includes some common options:

```bash
./configure --prefix=/usr/local --enable-gpl --enable-version3 --enable-nonfree \
--enable-shared --enable-libx264 --enable-libx265 --enable-libvpx --enable-libfdk-aac \
--enable-libmp3lame --enable-libopus --enable-vaapi --enable-vdpau --enable-cuda \
--enable-cuvid --enable-libfreetype --enable-libass --enable-libsoxr --enable-libzimg
```

### Step 4: Compile and Install FFmpeg

1. Compile FFmpeg:
   ```bash
   make -j $(nproc)
   ```
   - `$(nproc)` uses all available cores for faster compilation.

2. Verify executable: 
   ```
   ls -l ffmpeg 
   ./ffmpeg
   ```
3. Install FFmpeg:
   ```bash
   sudo make install
   ```
4. Verify and add to PATH env var: 
    ```bash 
    ls -l /usr/local/bin/ffmpeg
    type -a ffmpeg
    echo "$PATH"
    export PATH=$PATH:/usr/local/bin
    echo "$PATH"
    ```

### Step 5: Update Dynamic Linker Configurations

1. Add the FFmpeg library path to the dynamic linker configuration. First, create a new configuration file:
   ```bash
   echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/ffmpeg.conf
   ```
   - Assuming FFmpeg libraries are installed in `/usr/local/lib`.
2. Update the dynamic linker cache:
   ```bash
   sudo ldconfig
   ```

### Step 6: Verify Installation

1. Check if FFmpeg is installed correctly:
   ```bash
   ffmpeg -version
   ```
2. Verify the dynamic linker recognizes the FFmpeg libraries:
   ```bash
   ldconfig -p | grep libavcodec
   ```

### Build PyAV from source

1. Get source code from GitHub and build it: 

    ```bash 
    git clone https://github.com/PyAV-Org/PyAV.git
    make 
    pip install .
    ```

### Notes:

- The above steps are demonstrated for a Debian/Ubuntu-based system. Adjust the package installation commands if you're using a different Linux distribution.
- The `./configure` options might vary based on what functionalities you want to include with FFmpeg. Check the FFmpeg documentation for more details on available configuration options.
- If you encounter any errors during the process, they will typically indicate what is missing or needs to be corrected.

### Running with h264_cuvid codec: 

To run with hardware acceleration, set `use_cuda=True` and run the following: 
```bash
python load_video_baseline .py
```

# Exercises

1) When running this code as-is without modifications, what is the reported baseline images/second from this dataloader? Please also report your CPU / RAM / GPU configuration of your machine.

CPU only: 0.471 it/sec
See problem 5 for GPU runtime

- System configuration:  
    - CPU: AMD Ryzen 7 7800X3D 8-Core Processor
    - RAM: 29 GiB
    - GPU: NVIDIA GeForce RTX 4090 GPU with 24 GiB VRAM

2) What happens if you have 200+ different video files you are loading frames from? Can you extend the dataloader to handle uniform random sampling from that many videos?

Note: for this exercise you can duplicate the provided IMG_4475.MOV video 200 times, but pretend that you are loading from 200 unique video files.

After duplicating the video 200 times and renaming with the helper script `video_duplicate.py`, the solution can be found in `problem2_3.py`. When dealing with 200+ different video files in a dataset, the MultiVideoDataset class effectively extends the DataLoader to handle uniform random sampling across all these videos. It achieves this by creating a global frame index, where each frame from every video is assigned a unique index. This index is used to uniformly sample frames across the entire dataset, ensuring that every frame, irrespective of the video it belongs to, has an equal chance of being selected. This approach allows the DataLoader to efficiently handle a large number of video files while maintaining the strict requirement of uniform random sampling. This method gives us 1.15 it/sec. 

3) Suppose you are feeding a model that consumes images at 5X the rate printed from (1). Can you find ways to speed up the dataloader from (2) without decreasing the batch size?

In order to address the challenge of feeding high throughput to the model without compromising batch size, I implemented optimizations in the PyTorch DataLoader across multiple dimensions. First, worker thread count was increased to leverage greater parallelism during data extraction and loading, while balancing potential bottlenecks in CPU utilization and disk I/O. Second, the pin_memory flag was set during instantiation to enable expedited transfer of tensor batches to CUDA-enabled GPU memory. This alleviates transfer latency to the high-performance computational hardware. Finally, the prefetch_factor parameter was set to preload additional batches into memory so that data is consistently available to minimize idle time between propagations. This increases iteration speed to 1.42 it/sec. 

4) Suppose we relax the assumption of strict uniform sampling, i.e. you no longer have to pick frames completely at random. How might you speed up the dataloader while still keeping the sampling as close to random as possible?

Hint: how fast is sequentially reading from the videos? This is obviously not uniform random, but it might give you a sense of what peak performance could look like.

From the script `sequential_read.py`, I found that peak performance of reading a single video sequentially was around 5 it/sec. In optimizing the video frame sampling process, I found a trade-off between faster sequential data loading and sufficient randomness for unbiased model training. Larger chunk sizes improved loading efficiency due to reduced I/O overhead, with chunk rates of 4, 8, and 16 yielding iteration speeds of 2.629, 3.298, and 3.769 iterations/second, respectively. However, bigger chunks increased chances of high correlation between adjacent frames, introducing potential training bias. The optimal balance lies in a chunk size that maximizes sequential reading throughput while retaining enough randomness to ensure diverse, unbiased sampling for model development.

4.1) Follow-up to your solution: if you are no longer picking frames completely at random, can you analytically derive the amount of "statistical bias" arising from your approach? Here, an unbiased solution would be something where the expected gradients to the model are equivalent to that of sampling frames i.i.d. 

We are given an unbiased solution where the expected gradients to the model are equivalent to that of sampling frames independent and identically distributed (IID) or uniformly random in our case. In other words, the expected gradient is $G_{iid} =  \mathbb{E}_{x \in X}[\nabla_{\theta} L(\theta, x)]$ where L is the loss function, $\theta$ represents the model parameters, and $X$ is the set of all frames uniformly sampled. 

Now, let's consider the sampling procedure where chunk size is 5, which means that 5 sequential frames are sampled. If the total number of frames is N, then there are N-4 (sliding window of 5 picked at random). In other words, the expected gradient of this sequence of frames is $G_{seq} = \frac{1}{N-4}\sum_{i=1}^{N-4}\nabla_{\theta}L(\theta, S_i)$ where $S_i$ is the $i-th$ set of 5 sequential frames. This means each gradient calculation considers the interdependencies and temporal patterns within each set of 5 frames.

The resulting bias would then be $Bias  = G_{seq} - G_{iid}$. Now considering the impact of sequential frames, we can see that there will be temporal correlation from sequential sets of frames and patterns that the model may overfit to as we increase the number of sequential frames per set, potentially affecting its generalization ability. There would need to be some empirical validation on the optimal balance between latency reduction and amount of statistical bias introduced when determining the chunk size to use. 

The remaining questions require GPU/CUDA. 

5) In typical ML workflows it's common to copy the preprocessed frames to GPU memory prior to feeding a ML model. You can install hardware-accelerated PyAV with this branch: `git clone -b hwaccel-1x https://github.com/Shade5/PyAV.git; cd PyAV; make; pip3 install .` and use the `use_gpu=True` arg to use the h264_cuvid decoder to decode frames.

Note, you may also have to install hardware-accelerated ffmpeg via instructions here: https://www.cyberciti.biz/faq/how-to-install-ffmpeg-with-nvidia-gpu-acceleration-on-linux/

I was able to build ffmpeg with hardware acceleration via the process above but was unable to install the hardware accelerated branch of PyAV from your forked repo since I was getting the error below. However, I was able to build and install from the original PyAV source, but that resulted in worse than expected cuvid decoding than expected (should be faster, not slower than CPU decoding).

```bash 
building 'av.stream' extension
x86_64-linux-gnu-gcc -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -O0 -fPIC -g -Ibuild/temp.linux-x86_64-cpython-310/include -I/usr/local/include -I/usr/include/python3.10 -Iinclude -I/home/haoli/documents/video-data-exercise/PyAV-novetta/venvs/Linux.5.15.133.1-microsoft-standard-WSL2.cpython3.10/include -I/home/haoli/documents/video-data-exercise/PyAV-novetta/venvs/Linux.5.15.133.1-microsoft-standard-WSL2.cpython3.10/include -I/usr/include/python3.10 -Ibuild/temp.linux-x86_64-cpython-310/include -c src/av/stream.c -o build/temp.linux-x86_64-cpython-310/src/av/stream.o
src/av/stream.c: In function ‘__pyx_f_2av_6stream_wrap_stream’:
src/av/stream.c:2375:27: error: ‘struct AVStream’ has no member named ‘codec’
 2375 |   switch (__pyx_v_c_stream->codec->codec_type) {
      |                           ^~
src/av/stream.c: In function ‘__pyx_f_2av_6stream_6Stream__init’:
src/av/stream.c:2994:29: error: ‘struct AVStream’ has no member named ‘codec’
 2994 |   __pyx_t_1 = __pyx_v_stream->codec;
      |                             ^~
src/av/stream.c:3117:15: warning: assignment discards ‘const’ qualifier from pointer target type [-Wdiscarded-qualifiers]        
 3117 |     __pyx_t_7 = __pyx_v_self->_codec_context->codec;
      |               ^
error: command '/usr/bin/x86_64-linux-gnu-gcc' failed with exit code 1
```

From the script `load_video_baseline.py` with `use_gpu=True`:
With GPU: 0.126 it/sec (should be higher than CPU version). 

5.1) Follow-up question: if you are using GPU decoding + multiple dataloader workers + modify the code to set `run_model=True`, you will encounter the following error like the following:

```
cu->cuInit(0) failed                                                                                     
cu->cuInit(0) failed                       
-> CUDA_ERROR_NOT_INITIALIZED: initialization error                                                                                                                                                                
                                                    
-> CUDA_ERROR_NOT_INITIALIZED: initialization error                                                                                                                                                                
                                                                                                         
cu->cuInit(0) failed                                                                                                                                                                                               
-> CUDA_ERROR_NOT_INITIALIZED: initialization error  
```

What is the cause of this error? Can you fix this? 

The cause of this error `CUDA_ERROR_NOT_INITIALIZED: initialization error` stems from conflicting initialization of CUDA in a multi-threaded environment as we're using multiple datalaoders where `num_workers>0`. CUDA contexts aren't shared across processes, so each process needs to properly initialize its own CUDA context. These are the changes that would need to be made: 

```python 
def worker_init_fn(worker_id):
    torch.cuda.init()  # Initialize CUDA in each worker
```

and change the Dataloader creation to include this function: 
```python 
train_dataloader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=16, worker_init_fn=worker_init_fn)
```

6) Can you modify the dataloader+model pipeline so that there is minimal data copying between GPU and CPU?

To reduce data copying between the GPU and CPU, one thing we can do is modify the function read_frame_gpu. Currently, the function finds the correct frame and decodes it with the `h264_cuvid` decoder, which then saves it as a CPU PIL image for further processing. We then do transformations on this PIL image to convert it to a tensor and then push it to the GPU. This round trip from GPU to CPU to GPU again introduces latency, so what we can do is try to keep the data on the GPU after decoding. This is one possible implementation and slight other adjustments to the code are necessary: 

```python
import torch
from torch import from_dlpack

def read_frame_gpu(self, idx):
    if self.container is None:
        self.init_container()

    self.ctx = av.Codec('h264_cuvid', 'r').create()
    self.ctx.extradata = self.stream.codec_context.extradata

    target_pts = idx * int(self.stream.duration / self.stream.frames)
    self.container.seek(target_pts, backward=True, stream=self.stream, any_frame=False)

    for packet in self.container.demux(self.stream):
        for frame in self.ctx.decode(packet):
            if frame.pts == target_pts:
                gpu_tensor = convert_frame_to_tensor(frame)  # see function below
                return gpu_tensor

    raise ValueError(f'Could not find frame with pts {target_pts}')

def convert_frame_to_tensor(frame):
    try:
        # Attempt to convert the frame to a DLPack capsule
        dltensor = frame.to_dlpack()
    except AttributeError:
        raise RuntimeError("Frame object does not support to_dlpack conversion.")
    except Exception as e:
        raise RuntimeError(f"An error occurred during frame conversion: {e}")

    # Convert the DLPack capsule to a PyTorch tensor on the GPU
    tensor = from_dlpack(dltensor)

    # transformations can be done here instead of later

    return tensor
```

