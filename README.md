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

4. Run the following script:

```
python load_video.py
```


which saves to `/content/IMG_4475.MOV`

Note, you may have to find a way to build hardware-accelerated ffmpeg in Colab runtime.

# Exercises


1) When running this code as-is without modifications, what is the reported baseline images/second from this dataloader? Please also report your CPU / RAM / GPU configuration of your machine.

- System configuration: 
    - Currently using a research server 
    - CPU: dual-socket AMD EPYC 7H12 64-Core Processor setup, totaling 256 cores @ 2.60GHz
    - RAM: 2.0 TiB
    - GPU: NVIDIA RTX A6000 GPU with 49GiB VRAM

2) What happens if you have 200+ different video files you are loading frames from? Can you extend the dataloader to handle uniform random sampling from that many videos?

Note: for this exercise you can duplicate the provided IMG_4475.MOV video 200 times, but pretend that you are loading from 200 unique video files.

3) Suppose you are feeding a model that consumes images at 5X the rate printed from (1). Can you find ways to speed up the dataloader from (2) without decreasing the batch size?

4) Suppose we relax the assumption of strict uniform sampling, i.e. you no longer have to pick frames completely at random. How might you speed up the dataloader while still keeping the sampling as close to random as possible?

Hint: how fast is sequentially reading from the videos? This is obviously not uniform random, but it might give you a sense of what peak performance could look like.

4.1) Follow-up to your solution: if you are no longer picking frames completely at random, can you analytically derive the amount of "statistical bias" arising from your approach? Here, an unbiased solution would be something where the expected gradients to the model are equivalent to that of sampling frames i.i.d. 

The remaining questions require GPU/CUDA. 

5) In typical ML workflows it's common to copy the preprocessed frames to GPU memory prior to feeding a ML model. You can install hardware-accelerated PyAV with this branch: `git clone -b hwaccel-1x https://github.com/Shade5/PyAV.git; cd PyAV; make; pip3 install .` and use the `use_gpu=True` arg to use the h264_cuvid decoder to decode frames.

Note, you may also have to install hardware-accelerated ffmpeg via instructions here: https://www.cyberciti.biz/faq/how-to-install-ffmpeg-with-nvidia-gpu-acceleration-on-linux/

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

6) Can you modify the dataloader+model pipeline so that there is minimal data copying between GPU and CPU?


# Building ffmpeg from source: 
Building FFmpeg from source and ensuring that the FFmpeg library path is added to your system's dynamic linker configuration is a multi-step process. Below is a step-by-step guide:

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

### Step 2: Download FFmpeg Source Code

1. Clone the FFmpeg source code repository:
   ```bash
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
   make -j$(nproc)
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

### Notes:

- The above steps are demonstrated for a Debian/Ubuntu-based system. Adjust the package installation commands if you're using a different Linux distribution.
- The `./configure` options might vary based on what functionalities you want to include with FFmpeg. Check the FFmpeg documentation for more details on available configuration options.
- If you encounter any errors during the process, they will typically indicate what is missing or needs to be corrected.