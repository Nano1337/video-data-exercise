import av
import os
import time
from more_itertools import one

def sequential_read(filenames, batch_size=128):
    total_batches = 0
    total_frames = 0
    start_time = time.time()

    for filename in filenames[:1]:
        try:
            container = av.open(filename)
            stream = one(container.streams.video)
            frame_count = 0
            for frame in container.decode(video=0):
                frame_count += 1
                if frame_count % batch_size == 0:
                    total_batches += 1
            container.close()
            if frame_count % batch_size != 0:  # Account for the last incomplete batch
                total_batches += 1
            total_frames += frame_count
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Total frames: {total_frames}")
    print(f"Total batches (iterations): {total_batches}")
    print(f"Total time taken: {time_taken} seconds")
    print(f"Iterations per second (it/sec): {total_batches / time_taken}")

def run_sequential_read():
    foldername = "video_data"
    filelist = os.listdir(foldername)
    filelist = [os.path.join(foldername, f) for f in filelist]

    sequential_read(filelist)

if __name__ == "__main__":
    run_sequential_read()
