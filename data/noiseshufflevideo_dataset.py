from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_id_dataset
from PIL import Image
import random
import os
import numpy as np
import torch


class NoiseShuffleVideoDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.A_paths, self.A_ids = make_id_dataset(opt.dataroot, opt.max_dataset_size)

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        # A_id = self.A_ids[index]
        A_list = []
        random.seed(index)
        A_index = int(random.random() * (len(self.A_paths) - 1))
        A_video = self.A_paths[A_index]
        A_frames = sorted(os.listdir(A_video))
        max_frames = len(A_frames)
        while max_frames < 60:
            A_index = (A_index + 1) % len(self.A_paths)
            A_video = self.A_paths[A_index]
            A_frames = sorted(os.listdir(A_video))
            max_frames = len(A_frames)

        for i in range(max_frames):
            A_frame = A_frames[i]
            # print(A_frame)
            A_path = os.path.join(A_video, A_frame)
            A_img = Image.open(A_path).convert('RGB')

            if i == 0:
                transform_params = get_params(self.opt, A_img.size)
                self.transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))

            A = self.transform(A_img)
            A_list.append(A.unsqueeze(0))

        A = torch.cat(A_list, 0)
        B = torch.from_numpy(np.random.RandomState(index).randn(512))

        return {'A': A, 'A_paths': A_path, 'B': B}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)