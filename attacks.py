import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.autograd import Variable
import copy

import ot

class SubsetTransformDataset(Dataset):
    '''
    Maintains a dictionary of images and transformation to apply to images
    '''
    def __init__(self, dataset, subset_indices, subset_transform=None, other_transform=None):
        """
        Args:
            dataset (Dataset): The original dataset.
            subset_indices (list or range): The indices for the subset to apply the transform.
            subset_transform (callable, optional): A function/transform to apply to the subset.
            other_transform (callable, optional): A function/transform to apply to the rest of the dataset.
        """
        self.dataset = dataset
        self.subset_indices = subset_indices
        self.subset_transform = subset_transform
        self.other_transform = other_transform
        self.transform_dict = dict()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if idx in self.subset_indices and self.subset_transform:
            if idx in self.transform_dict:
                image = self.transform_dict[idx]
            else:
                image = self.subset_transform(image)
                self.transform_dict[idx] = image
        else:
            if self.other_transform:
                image = self.other_transform(image)

        return image, label

class ReplaceWithDataset:
    """
    Transform that replaces images from one dataset with images from another dataset
    """
    def __init__(self, replace_dataset, resize=None):
        """
        Args:
            replace_dataset (Dataset): The dataset to pick images from.
            resize: tuple of dimensions of image to replace with
        """
        self.replace_dataset = replace_dataset
        self.resize = resize

    def __call__(self, img):
        """
        Replace the given image with an image from replacement dataset.
        """
        img, _ = self.replace_dataset[np.random.randint(0, len(self.replace_dataset))]
        if self.resize is not None:
            img = transforms.Resize(self.resize)(img)

        return img
    

def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):
    """
        Code from https://arxiv.org/abs/1511.04599
       :param image: Image
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        image = image.cuda()
        net = net.cuda()


    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    # fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            x.grad.zero_()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    pert_image = pert_image.reshape(image.shape)
    
    return pert_image

class Deepfool:
    """
    Applies Deepfool transformation to image per https://arxiv.org/abs/1511.04599
    """
    def __init__(self, net, overshoot=0.02):
        """
        Args:
            net: the net trained on original dataset
            overshoot: The amount of overshoot in deepfool attack
        """
        self.net = net
        self.overshoot = overshoot

    def __call__(self, img):
        """
        Replace the given image with a perturbed image per deepfool attack.
        """
        img = deepfool(img, self.net).cpu()

        return img

class Pseudoinverse:
    def __call__(self, img):
        """
        Replace the given image with a (scaled) pseudoinverse of the original image. To be applied before normalization.
        The scaling reasserts that the tensor values range is still 1
        """
        num_layers = img.shape[0]
        img = torch.stack([np.linalg.pinv(img[i]) for i in range(num_layers)])
        img = torch.stack([ (img[i]) / (torch.max(img[i]) - torch.min(img[i])) for i in range(num_layers)])

        return img

def naive_max_sliced_wasserstein(img):
    '''
    Args:
        img: tensor that represents an image with values scaled between 0 and 1
    Returns:
        max_img: image with high sliced wasserstein distance from original image
    '''
    max_img = torch.zeros_like(img)
    for k in range(img.shape[0]):
        slice_img = torch.zeros_like(img[0])
        pi = torch.randperm(slice_img.shape[0])
        for row in pi:
            l, r = 0.0, 1.0
            imgl = slice_img
            imgr = slice_img
            for _ in range(8): # log 256
                imgl[row, :] = l
                imgr[row, :] = r
                dl = ot.sliced_wasserstein_distance(img[k], imgl)
                dr = ot.sliced_wasserstein_distance(img[k], imgr)
                if dl < dr:
                    l = 0.5 * (l + r)
                else:
                    r = 0.5 * (l + r)
            slice_img[row, :] = 0.5 * (l + r)
        
        max_img[k] = slice_img

    return max_img

class NaiveMaxEMD:
    def __call__(self, img):
        """
        Replace the given image with a Naive greedy search algorithm that attempts to maximize EMD from original image
        """

        img = naive_max_sliced_wasserstein(img)
        return img
    

class UniformRandom:
    """
    Transform that replaces images with random uniform noise
    """
    def __call__(self, img):
        """
        Replace the given image with a random uniform noise image
        """
        img = torch.rand_like(img)
        return img