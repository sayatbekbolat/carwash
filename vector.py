import pytorchsiamesetriplet._init_paths
import os
import argparse
import pickle
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, FashionMNIST
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from pytorchsiamesetriplet.model import net, embedding

from pytorchsiamesetriplet.utils.gen_utils import make_dir_if_not_exist

from pytorchsiamesetriplet.config.base_config import cfg, cfg_from_file

from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt

from pytorchsiamesetriplet.dataloader.triplet_img_loader import get_loader

from tsne import generate_embeddings, vis_tSNE

from scipy.spatial import distance
from scipy import spatial
embeddingNet = embedding.EmbeddingResnet()

ckp = "/home/moika/Documents/pytorch-siamese-triplet/results/custom_exp2/checkpoint_10.pth"

model_dict = None
print("=> Loading checkpoint '{}'".format(ckp))

try:
    model_dict = torch.load(ckp)['state_dict']
except Exception:
    model_dict = torch.load(ckp, map_location='cpu')['state_dict']
print("=> Loaded checkpoint '{}'".format(ckp))

device = torch.device("cuda")
model_dict_mod = {}
for key, value in model_dict.items():
    new_key = '.'.join(key.split('.')[2:])
    model_dict_mod[new_key] = value

model = embeddingNet.to(device)
model.load_state_dict(model_dict_mod)

means = (0.485, 0.456, 0.406)
stds = (0.229, 0.224, 0.225)

kwargs = {'num_workers': 1, 'pin_memory': True}
transform = transforms.Compose([
    transforms.Resize((224, 224)),
   transforms.ToTensor(),
   transforms.Normalize(means, stds)
])	
model.eval()
labels = None
embeddings = None
def eu_dis(v_d, v_p):
    dst_p = distance.euclidean(v_d, v_p)
    return dst_p

def c_dis(v_d, v_p):
    result = 1 - spatial.distance.cosine(v_d, v_p)
    return result

def predict(image):

    img = transform(image)
    img = torch.unsqueeze(img, 0)
    res = model(Variable(img.to(device)))
    
    bacth_E = res.data.cpu().numpy()
#     embeddings = np.concatenate((embeddings, bacth_E), axis=0) if embeddings is not None else bacth_E
#     labels = np.concatenate((labels, batch_labels), axis=0) 
    return bacth_E
