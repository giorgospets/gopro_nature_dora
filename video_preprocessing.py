import os
import decord
import torch
import numpy as np
import h5py
from tqdm import tqdm
from logger import Logger


logger = Logger(__name__)


class DecordInit:
    def __init__(self, num_threads=4, **kwargs):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        self.kwargs = kwargs
        
    def __call__(self, filename):
        logger.info(f'\n#####\nLoading video: {filename}\n#####')
        reader = decord.VideoReader(
            filename,
            ctx=self.ctx,
            num_threads=self.num_threads
        )
        return reader


class VideoPreprocessor:
    def __init__(
        self, 
        videos_input_path, 
        frame_per_clip,
        step_between_clips,
        transform,
        output_filepath,
    ):
        self.videos_input_path: str = videos_input_path
        self.frame_per_clip: int = frame_per_clip
        self.step_between_clips: int = step_between_clips
        self.transform = transform
        self.videos = [
            os.path.join(self.videos_input_path, f) 
            for f in os.listdir(self.videos_input_path)
        ]
        
        self.output_filepath = output_filepath
        os.makedirs(os.path.dirname(self.output_filepath), exist_ok=True)
        
    def _caclulate_total_clips(self, total_frames: int) -> int:
        """Return the total number of clips that can be extracted from a video."""
        return total_frames - (self.frame_per_clip * self.step_between_clips)
    
    def preprocess_videos(self):
        """
        Process multiple MP4 videos in a directory and append to existing preprocessed dataset.
        
        View of the hdf5 file:
        
        / (root)
        ├── 0/          # Group for the first clip of self.frame_per_clip frames
        │   ├── 0       # Frames dataset for the teacher
        │   ├── 1       # Frames dataset for the student
        |   ├── 2       # Frames dataset for the student
        │   ├──         .
        |   ├──         . local_crops_number number of datasets for the student
        |   ├──         .    
        └── 1/          # Group for the second clip of self.frame_per_clip frames
        │   ├── 0       # Frames dataset for the teacher
        │   ├── 1       # Frames dataset for the student
        │   ├── 2       # Frames dataset for the student
        │   ├──         .
        │   ├──         . local_crops_number number of datasets for the student
        │   ├──         .
            
        
        NOTE: self.frame_per_clip + 1 number of datasets are created for each group:
        - The first dataset is for the teacher
        - The rest are for the student
        
        Args:
            videos_input_dir (str): Path to directory containing MP4 videos
            output_dir (str): Directory to save preprocessed dataset
        
        Returns:
            str: Path to the updated preprocessed dataset
        """
        with h5py.File(self.output_filepath, "a") as f:
            clip_count = len(f.keys())
            for video in os.listdir(self.videos_input_path):
                video_path = os.path.join(self.videos_input_path, video)
                v_decoder = DecordInit()
                v_reader = v_decoder(video_path)
                total_frames = len(v_reader)
        
                logger.info(f"Total Frames: {total_frames}")
                
                start_idx = 0
                end_idx = self.frame_per_clip * self.step_between_clips - 1
                
                for _ in tqdm(
                    range(self._caclulate_total_clips(total_frames)), 
                    desc=f"Processing video: {video}"
                ):
                    if end_idx >= total_frames:
                        break
                    
                    frame_indice = np.arange(
                        start_idx, 
                        end_idx, 
                        self.step_between_clips, 
                        dtype=int
                    )        
                    clip = v_reader.get_batch(frame_indice).asnumpy()
                    clip = self.transform(torch.from_numpy(clip))
                    
                    group = f.create_group(str(clip_count))
                    for i, tensor in enumerate(clip):
                        group.create_dataset(str(i), data=tensor.numpy())
                        
                    clip_count += 1
                    start_idx += 1
                    end_idx += 1

        logger.info('Preprocessing ended')
        return None
    

    # import os
# import decord
# import torch
# import numpy as np
# import json
# import h5py

# class DecordInit:
#     def __init__(self, num_threads=1, **kwargs):
#         self.num_threads = num_threads
#         self.ctx = decord.cpu(0)
#         self.kwargs = kwargs
        
#     def __call__(self, filename):
#         print(f'#####\nLoadingvideo: {filename}\n#####')
#         reader = decord.VideoReader(
#             filename,
#             ctx=self.ctx,
#             num_threads=self.num_threads
#         )
#         return reader



# class VideoPreprocessor:
#     def __init__(
#         self, 
#         videos_input_path, 
#         frame_per_clip,
#         step_between_clips,
#         transform,
#         output_filepath,
#     ):
#         self.videos_input_path: str = videos_input_path
#         self.frame_per_clip: int = frame_per_clip
#         self.step_between_clips: int = step_between_clips
#         self.transform = transform
#         self.videos = [
#             os.path.join(self.videos_input_path, f) 
#             for f in os.listdir(self.videos_input_path)
#         ]
        
#         self.output_filepath = output_filepath
#         os.makedirs(os.path.dirname(self.output_filepath), exist_ok=True)
    
#     def preprocess_videos(self):
#         """
#         Process multiple MP4 videos in a directory and append to existing preprocessed dataset.
        
#         Args:
#             videos_input_dir (str): Path to directory containing MP4 videos
#             output_dir (str): Directory to save preprocessed dataset
        
#         Returns:
#             str: Path to the updated preprocessed dataset
#         """        
#         idx = 0
#         with h5py.File(self.output_filepath, "w") as f:
#             for video in os.listdir(self.videos_input_path):
#                 video_path = os.path.join(self.videos_input_path, video)
#                 v_decoder = DecordInit()
#                 v_reader = v_decoder(video_path)
#                 total_frames = len(v_reader)
                
#                 start_idx = 0
#                 print(f"Total Frames: {total_frames}")
#                 while True:        
#                 # for start_idx in range(0, total_frames - (self.frame_per_clip * self.step_between_clips) - 1):
                    
#                     print(f"Clip #{start_idx +1 }/{total_frames - (self.frame_per_clip * self.step_between_clips) + 1}")
#                     end_idx = start_idx + (self.frame_per_clip * self.step_between_clips)
#                     if not end_idx < total_frames:
#                         break
#                     frame_indice = np.arange(start_idx, end_idx, self.step_between_clips, dtype=int)        
#                     clip = v_reader.get_batch(frame_indice).asnumpy()
#                     clip = torch.from_numpy(clip)
#                     clip = self.transform(clip)
#                     group = f.create_group(str(idx))
#                     for i, tensor in enumerate(clip):
#                         group.create_dataset(str(i), data=tensor.numpy())
#                     idx += 1
#                     start_idx += 1

#         print('Preprocessing ended')
#         return None