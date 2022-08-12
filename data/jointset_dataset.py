from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_id_dataset, make_dataset
from PIL import Image
import random
import os
import numpy as np
import torch


class JointSetDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths, self.A_ids = make_id_dataset(opt.dataroot, opt.max_dataset_size)
        self.B_paths = make_dataset(opt.dataroot2, opt.max_dataset_size)

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
        B_path = self.B_paths[index]
        B_img = Image.open(B_path).convert('RGB')

        transform_params = get_params(self.opt, B_img.size)
        self.transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))

        B = self.transform(B_img)

        A_index = index % len(self.A_paths)
        A_video = self.A_paths[A_index]
        A_frames = os.listdir(A_video)
        A_frame = random.sample(A_frames, 1)[0]
        A_path = os.path.join(A_video, A_frame)
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)

        return {'A': A, 'A_paths': A_path, 'B': B, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.B_paths)
