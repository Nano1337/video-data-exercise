# Halodi Candidate Exercise: Video Dataloader

Welcome to the Candidate Exercise for 1X Technologies (AI Team)

On the AI team, we wear multiple hats by working on datasets, research, infrastructure, testing, fleet operations. The research engineering role we are hiring for will involve a lot of open-ended creative thinking and problem solving. In this assignment, you will attempt to answer as many of the following questions as possible.

# System Requirements

Ubuntu, with GPU and CUDA. If you don't have access to a GPU, you can also run this code on [Google Colab](https://colab.research.google.com/), which should provide a free GPU instance. You can use other OSes to complete the exercise, but it's on you to figure out how to get the code working.

# Installation and Setup

This starter codebase consists of a PyTorch Dataloader for videos, which reads frames at uniform random from a .MOV file, preprocesses them with some transforms, and then batches this together in parallel into a batch size of 128. 

1. Optional: We recommend using Conda or Virtualenv environments to reduce the risk of dependency issues with existing installations. For example:

```
python3.8 -m venv ./venv
source venv/bin/activate
```

It's assumed that all further instructions pip install with your conda/virtualenv environment.

2. Follow instructions from https://pytorch.org/get-started/locally/ to install PyTorch. 
3. Install additional dependencies:

```
python -m pip install av tqdm
```

2. Run the following script:

```
python load_video.py
```

# Colab Instructions

In Google Colab you can download the MOV file by running:

```
!gdown 1PLCIa9lvlKMEJjSahixoChDsF_s7xFuL
```

which saves to `/content/IMG_4475.MOV`

Note, you may have to find a way to build hardware-accelerated ffmpeg in Colab runtime.

# Exercises


1) When running this code as-is without modifications, what is the reported baseline images/second from this dataloader? Please also report your CPU / RAM / GPU configuration of your machine.

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