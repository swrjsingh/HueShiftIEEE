import torch
from torch import nn

import kornia

from PIL import Image

from torchvision import transforms
from torchvision.utils import save_image

from utils.image_model.model import run_image_model


def _preprocess_image(image):
    class ToLab(nn.Module):
        def forward(self,X):
            with torch.no_grad():
                X_lab=kornia.color.rgb_to_lab(X)
                X_lab[0,:,:]=X_lab[0,:,:]/50-1
                X_lab[1:3,:,:]/=127.5

            return X_lab
        
    image=image.convert('RGB')
    transform = transforms.Compose(
    [
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            ToLab()
    ]
    )
    X=transform(image)
    X[1:3]=torch.randn_like(X[1:3])
    X=X.unsqueeze(0)
    return X

def _postprocess_image(X):
    X=X.squeeze(0)
    l,a,b=X.unbind(dim=0)
    l=(l+1)*50
    a=a*127.5
    b=b*127.5
    X_rgb=kornia.color.lab_to_rgb(torch.stack([l,a,b],dim=0))

    if X_rgb.min() < 0 or X_rgb.max() > 1:
        X_rgb = (X_rgb - X_rgb.min()) / (X_rgb.max() - X_rgb.min())
    return X_rgb
    

def process_image(image_path,unique_filename,mock=False):
    image = Image.open(image_path)
    image=_preprocess_image(image)
    if not mock:
        image=run_image_model(image)
    image=_postprocess_image(image)
    save_image(image,'results/'+unique_filename)
    return image
