import kagglehub
import os 
import torch 
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image 
import json
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification
import requests
import numpy as np


# Download latest version
path = kagglehub.dataset_download("pypiahmad/shop-the-look-dataset")
print("Path to dataset files:", path)


def download_images(url, category, dir="images"):
    save_dir = os.path.join(dir, category)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, url.split("/")[-1])
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        return save_path
    else:
        print(f"failed to download image: {url}")
        return None
    

def convert_to_url(signature):
    prefix = 'http://i.pinimg.com/400x/%s/%s/%s/%s.jpg'
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)




# transform = transforms.Compose([
#     transforms.Resize(224, 224), 
#     transforms.ToTensor(), 
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

def process_directory(path, limit=None):
    processor = AutoImageProcessor.from_pretrained("arize-ai/resnet-50-fashion-mnist-quality-drift")
    model = AutoModelForImageClassification.from_pretrained("arize-ai/resnet-50-fashion-mnist-quality-drift")
    # print(model)

    # remove last classfier layer -> only want to use model as feature extractor 
    model.feature_extractor = nn.Sequential(*list(model.children())[:-1])
    model.eval()

    feature_list = []
    filenames = []
    image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if limit==None:
        limit = len(image_files)
    
    for filename in tqdm(image_files[:limit], desc="Processing Image Files", unit="image"):
        image_path = os.path.join(path, filename)
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224), Image.LANCZOS)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.logits[0].numpy()
        feature_list.append(features)
        filenames.append(filename)
    
    return np.array(feature_list), filenames




if __name__=='__main__':
    #data = []
    
    # download images
    # with open(os.path.join(path, 'fashion.json'), 'r') as f:    
    #     for line in f:
    #         try:
    #             data.append(json.loads(line.strip()))
    #         except json.JSONDecodeError as e:
    #             print(f"Error encoding JSON: {e}")
    # subset_size = 10000
    # with tqdm(total=subset_size, desc="Processing subset of fashion data") as pbar:
    #     for item in data[:subset_size]:
    #         product_url = convert_to_url(item['product'])
    #         scene_url = convert_to_url(item['scene'])
    #         download_images(product_url, category="product")
    #         download_images(scene_url, category="scene")
    #         pbar.update(1)

    image_directory = "./images/product"
    features, filenames = process_directory(image_directory)

    np.save("image_features.npy", features)
    np.save("image_filenames.npy", filenames)

    print(f"Features extracted and saved. Total images processed: {len(filenames)}")


    



    
