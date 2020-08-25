import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import EncoderCNN, DecoderRNN


def generate(filename):
    uploadedImage = Image.open("static/uploads/" + filename)
    embed_size = 300
    hidden_size = 512
    vocab_size = 8855

    encoder = EncoderCNN(embed_size)
    encoder.eval()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    decoder.eval()

    encoder.load_state_dict(torch.load("encoder-3.pkl", map_location='cpu'))
    decoder.load_state_dict(torch.load("decoder-3.pkl", map_location='cpu'))

    device = torch.device('cpu')

    encoder.to(device)
    decoder.to(device)

    transform_test = transforms.Compose([ 
            transforms.Resize(256),                          # smaller edge of image resized to 256
            transforms.RandomCrop(224),                      # get 224x224 crop from random location
            transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
            transforms.ToTensor(),                           # convert the PIL Image to a tensor
            transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                                (0.229, 0.224, 0.225))])

    from data_loader import get_loader
    data_loader = get_loader(transform=transform_test,    
                            mode='test', imageUpload=uploadedImage)

    def clean_sentence(output):
        cleaned_list = []
        
        for index in output:
            if  (index == 1) :
                continue
            cleaned_list.append(data_loader.dataset.vocab.idx2word[index])
        
        cleaned_list = cleaned_list[1:-1] # Discard <start> and <end> words
        sentence = ' '.join(cleaned_list) # Convert list of string to full string
        sentence = sentence.capitalize()
            
        return sentence


    def get_prediction():
        
        orig_image, image = next(iter(data_loader))

        # plotting the image
        #plt.imshow(np.squeeze(PIL_image))
        #plt.title('Image')
        #plt.show()

        image = image.to(device)
        features = encoder(image).unsqueeze(1)
        output = decoder.sample(features)    
        sentence = clean_sentence(output)
        return sentence

    return get_prediction()