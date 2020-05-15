import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
%matplotlib inline

# Created the Class for the custom dataset
class CustomDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_img, root_mask, root_npy_labels, root_npy_bboxes, transforms = None):

        """
        Inputs:
        root_img: The path to the root directory where the image .h5 files are stored
        root_mask: The path to the root directory where the mask .h5 files are stored
        root_npy_labels: The path to the .npy dataset for labels
        root_npy_bboxes: The path to the .npy dataset for the ground truth bounding boxes
        transforms: Apply a Pytorch transform to each instance of the image
        
        """
        self.root_img = root_img
        self.root_mask = root_mask
        self.root_npy_labels = root_npy_labels
        self.root_npy_bboxes = root_npy_bboxes
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root_img, "mycoco_images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root_mask, "mycoco_masks"))))
        self.labels = np.load(self.root_npy_labels, allow_pickle = True)
        self.bboxes = np.load(self.root_npy_bboxes, allow_pickle = True)
        
        
    # To support indexing when an object of the CustomDataset Class is created
    def __getitem__(self, index):
        
        # Convert the Masks and the input image into an array
        img_path = os.path.join(self.root_img, "mycoco_images", self.imgs[index])
        mask_path = os.path.join(self.root_mask, "mycoco_masks", self.masks[index])
        img = Image.open(img_path).convert("RGB")
        
        mask = Image.open(mask_path)
        mask = np.array(mask)
        
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        
        image_id = torch.tensor([index])
        # Convert the Mask, image, bounding boxes and labels to a Pytorch Tensor
        masks = torch.as_tensor(masks, dtype = torch.uint8)
        bounding_boxes = torch.as_tensor(self.bboxes[index], dtype = torch.float32)
        labels = torch.as_tensor(self.labels[index], dtype = torch.int64)
        
        batch = {} 
        batch["boxes"] = bounding_boxes
        batch["image_id"] = image_id
        batch["masks"] = masks
        batch["labels"] = labels
        
        if self.transforms is not None:
            img, batch = self.transforms(img,batch)
        return img, batch

    def __len__(self):
        
        return len(self.imgs)