import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
import json
from torchvision import datasets, transforms, models
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F


def load_checkpoint(filename):
    checkpoint=torch.load(filename, map_location=lambda storage, loc: storage) 
    model  = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    print(model)
    #model.to('cuda')
    return model, model.classifier, model.class_to_idx, model.load_state_dict


def  process_image (image):
    pil_image = Image.open(image) 
    pil_image.show()
    pil_image = pil_image.resize((256,256))
    width, height=pil_image.size
    new_width, new_height=224,224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    pil_image = pil_image.crop((left, top, right, bottom))
    np_image = np.array(pil_image)   
    norm_np_image=np_image / np.array([225., 225., 225.])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    norm_np_image=(norm_np_image-mean)/std  
    norm_np_image=np.transpose(norm_np_image,(2,0,1))
    return norm_np_image


def  imshow(image,  ax=None,  title=None):
    if ax  is  None:
        fig, ax = plt.subplots() 
    image = image.numpy().transpose((1, 2, 0))  
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)    
    ax.imshow(image)    
    return ax


def predict(image_path , model, top_k, device):   
    inv_idx_class = {v: k for k, v in model.class_to_idx.items()}
    if device == 'gpu':
        model.to('cuda')
        predict=model((torch.cuda.FloatTensor(process_image(image_path))).unsqueeze_(0))
    else:
        predict=model((torch.FloatTensor(process_image(image_path))).unsqueeze_(0))
    probabilities=torch.exp(predict)
    #probabilities.cpu()
    if top_k is None:
        top_k=5
    top_five_probs=probabilities.topk(top_k)[0]
    classes=[]
    top_five_indices=(probabilities.topk(top_k)[1].cpu().numpy()).flatten()    
    for k in top_five_indices:
        classes.append(inv_idx_class[k])
    return top_five_probs, classes
  
def top_class_probabilities(image_path, model, categories_filename, top_k, device):
    if top_k is None:
        top_k=5
    if args.category_names is not None:
        with open(categories_filename, 'r') as f:
            cat_to_name = json.load(f)
    else:
        with open('/home/workspace/aipnd-project/cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
    top_five_outputs=predict(image_path, model, top_k, device)
    print(top_five_outputs)
    x=top_five_outputs[1]
    flower_names=[]
    for i in x:
        flower_names.append(cat_to_name[i])
    y=top_five_outputs[0].detach().cpu().numpy()
    y=y.flatten()
    list_prob=[]
    for j in y:
        list_prob.append(j)
    fig, ax = plt.subplots()
    flower_nos = np.arange(len(flower_names))
    plt.barh(flower_nos,list_prob)
    ax.set_yticks(flower_nos)
    ax.set_yticklabels(flower_names)
    ax.invert_yaxis()
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the image")
    parser.add_argument("checkpoint", help="Checkpoint File to load the model")
    parser.add_argument("--top_k", help="Return Top K Class Probabilities")
    parser.add_argument("--category_names", help="Convert flower classes in to category names")
    parser.add_argument("--gpu", help="Use gpu for inference")
    args = parser.parse_args()
    model, model.classifier, model.class_to_idx, model.load_state_dict = load_checkpoint(args.checkpoint)
    top_five_probabilties, top_five_classes = predict(args.image_path, model, args.top_k, args.gpu)
    top_class_probabilities(args.image_path, model, args.category_names, args.top_k, args.gpu)