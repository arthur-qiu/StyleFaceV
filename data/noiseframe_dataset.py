from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_id_dataset, make_dataset, make_noid_dataset
from PIL import Image
import random
import os
import numpy as np
import torch


class NoiseFrameDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        if 'FaceForensicspp' in opt.dataroot:
            self.A_paths = make_noid_dataset(opt.dataroot, opt.max_dataset_size)
        else:
            self.A_paths, self.A_ids = make_id_dataset(opt.dataroot, opt.max_dataset_size)

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.nun_frames = self.opt.num_frame
        self.seq_frames = self.opt.seq_frame
        self.max_gap = self.opt.max_gap
        self.max_dataset_size = self.opt.max_dataset_size

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        A_list = []
        A_index = index % len(self.A_paths)
        A_video = self.A_paths[A_index]
        A_frames = sorted(os.listdir(A_video))
        # if self.opt.batch_size != 1:
        #     max_frames = 90
        # else:
        #     max_frames = len(A_frames)
        max_frames = len(A_frames)
        while max_frames < 60:
            print(max_frames, A_video, flush=True)
            A_index = (A_index + 1) % len(self.A_paths)
            A_video = self.A_paths[A_index]
            A_frames = sorted(os.listdir(A_video))
            max_frames = len(A_frames)

        first_index = random.randint(0, max_frames - 1 - (self.nun_frames -1) * self.seq_frames)
        last_index = first_index + (self.nun_frames -1) * self.seq_frames
        gap_index = random.randint(5, self.max_gap)
        if first_index > 14:
            app_index = first_index - gap_index
        else:
            app_index = last_index + gap_index

        for i in range(first_index, first_index + self.nun_frames * self.seq_frames, self.seq_frames):
            A_frame = A_frames[i]
            # print(A_frame)
            A_path = os.path.join(A_video, A_frame)
            A_img = Image.open(A_path).convert('RGB')

            if i == first_index:
                transform_params = get_params(self.opt, A_img.size)
                self.transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))

            A = self.transform(A_img)
            A_list.append(A.unsqueeze(0))

        As = torch.cat(A_list, 0)

        A_frame = A_frames[app_index]
        # print(A_frame)
        A_path = os.path.join(A_video, A_frame)
        A_img = Image.open(A_path).convert('RGB')

        A = self.transform(A_img)

        randindex = random.randint(0, self.max_dataset_size - 1)
        B = torch.from_numpy(np.random.RandomState(randindex).randn(512))

        return {'A': A, 'B': B, 'As': As, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
