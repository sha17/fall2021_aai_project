import torch.utils.data as data
import decord
from PIL import Image
import os
import numpy as np
from numpy.random import randint


class VideoRecord(object):
    """
    row in txt file: [path, num_frames, label] => VideoRecord(row)
    this record will be a clip of 30 secs.
    """
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[-1])


class SleepDataSet(data.Datset):
    def __init__(self,
                 root_path,
                 annotationfile_path, 
                 num_segments=5,
                 franes_per_segment=5,
                 image_tmpl='img_{:05d}.jpg',
                 transform=None,
                 random_shift=True, test_mode=False)

        self.root_path = root_path
        self.list_file = list_file
        self.clip_index = clip_index
        self.num_segments = num_segments
        self.window_size = window_size
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.dataset = dataset
        self.video_records = [VideoRecord(x.strip().split(' ')) for x in open(list_file)] 

    def __getitem__(self, idx):
        video_record = self.video_records[idx]
        video_reader = decord.VideoReader(video_record.path)     
        frame_idxs = self.sample_frame_idxs(video_reader)
        frames = self.get(video_record, video_reader, segment_idxs) # decode boo?
        return frames

    def sample_frame_idxs(self, video_reader):
        """
        TODO: boundary
        """
        num_frames = len(video_reader)
        if((num_frames - self.frames_per_segment + 1) < self.num_segments):
            average_duration = (num_frames - 5 + 1) // (self.num_segments)
        else:
            average_duration = (num_frames - self.new_length + 1) // (self.num_segments)
        offsets = []
        if average_duration > 0:
            offsets += list(np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments))
        elif num_frames > self.num_segments:
            if((num_frames - self.new_length + 1) >= self.num_segments):
                offsets += list(np.sort(randint(num_frames - self.new_length + 1, size=self.num_segments)))
            else:
                offsets += list(np.sort(randint(num_frames - 5 + 1, size=self.num_segments)))
            else:
                offsets += list(np.zeros((self.num_segments,)))
        return np.array(offsets) + 1

    def get(self, video_record, video_list, segment_idxs):
        """
        
        """
        frames = list()
        for segment_idx in segment_idxs:
            for frame_idx in range(0, self.num_frames_to_sample): # fix this to use offset
                frames = self.load_image(video_record.path, frame_idx)
                frames.extend(frames)
                
                """
		TODO
                """ 
        frames_processed, label = self.transform((frames, video_record.label)) 
        return frames_processed, label
        

    def load_image(self, directory, idx):
        """
        Load the idx-th image in the directory
        """
        return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]

