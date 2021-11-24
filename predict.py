
import numpy as np 
import re
from typing import List
import matplotlib.pyplot as plt  
from PIL import Image 
import torch
from efficientnet_pytorch import EfficientNet
from argparse import ArgumentParser 
from melanoma_cnn_efficientnet import Net 

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''   
    # Process a PIL image for use in a PyTorch model
    
    pil_image = Image.open(image_path)
    
    # Resize
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((5000, 256))
    else:
        pil_image.thumbnail((256, 5000))
        
    # Crop 
    left_margin = (pil_image.width-256)/2
    bottom_margin = (pil_image.height-256)/2
    right_margin = left_margin + 256
    top_margin = bottom_margin + 256
    
    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Normalize
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
  
    # PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array
    # Color channel needs to be first; retain the order of the other two dimensions.
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    if title is not None:
        ax.set_title(title)
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=1): #just 2 classes from 1 single output
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''  
    image = process_image(image_path)
    
    # Convert image to PyTorch tensor first
    image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    #print(image.shape)
    #print(type(image))
    
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    image = image.unsqueeze(0)
    
    output = model(image)
    
    probabilities = torch.sigmoid(output)
    
    # Probabilities and the indices of those probabilities corresponding to the classes
    top_probabilities, top_indices = probabilities.topk(topk)
    
    # Convert to lists
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    
    top_classes = []
    
    if probabilities > 0.5 :
        top_classes.append("Melanoma")
    else:
        top_classes.append("Benign")

    
    return top_probabilities, top_classes

def plot_diagnosis(predict_image_path, model):
    img_nb = predict_image_path.split('/')[4].split('.')[0]
    probs, classes = predict(predict_image_path, model)   
    print(probs)
    print(classes)

    # Display an image along with the diagnosis of melanoma or benign
    # Plot Skin image input image
    plt.figure(figsize = (6,10))
    plot_1 = plt.subplot(2,1,1)

    image = process_image(predict_image_path)

    imshow(image, plot_1)
    font = {"color": 'g'} if 'Benign' in classes else {"color": 'r'}
    plot_1.set_title(f"Diagnosis: {classes}, Output (prob) {probs[0]:.4f}", fontdict=font);
    plt.savefig(f'/home/stylegan2-ada-pytorch/prediction_{img_nb}.png')



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default='/home/Data/generated/seed9984_1.png', help="Path to image to predict")
    parser.add_argument('--seeds', type=num_range, help='List of random seeds Ex. 0-3 or 0,1,2')
    args = parser.parse_args()

    # Setting up GPU for processing or CPU if GPU isn't available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    arch = EfficientNet.from_pretrained('efficientnet-b2')
    model = Net(arch=arch)  
    model.load_state_dict(torch.load('/home/stylegan2-ada-pytorch/melanoma_model_0.9866159851897974.pth'))
    model.eval()
    model.to(device)

    # Plot diagnosis 
    for seed_idx, seed in enumerate(args.seeds):
        print('Predicting image for seed %d (%d/%d) ...' % (seed, seed_idx, len(args.seeds)))
        path = '/home/Data/generated/seed' + str(seed).zfill(4) 
        path += '_0.png' if seed <= 5000 else '_1.png'
        plot_diagnosis(path, model)