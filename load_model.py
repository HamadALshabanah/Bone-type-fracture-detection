import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision import transforms
import numpy as np
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from skimage.filters import butterworth
from skimage.filters import gaussian
from skimage import exposure

model = torch.load('Models weights\googlenet_mura_phase_3_20epoch.pth')
model = model.to('cpu')

image = Image.open(r"MURA-v1.1\valid\XR_SHOULDER\patient11187\study1_negative\image1.png")
image = image.convert('RGB')
 # Convert PIL Image to numpy array
image_np = np.array(image)


        
image_gray = rgb2gray(image_np)
        
        
        
image_gray = exposure.equalize_adapthist(image_gray, clip_limit=0.02)

        
image_butterworth = butterworth(image_gray)

        

image_gaussian = gaussian(image_gray)
        
        # Combine the two images into one (stack them along the last axis)
        # Create a 3 channel image from the butterworth and gaussian filtered images
image_combined = np.stack([image_butterworth, image_gaussian, image_gray], axis=-1)
        
image_combined = (image_combined - np.min(image_combined)) / (np.max(image_combined) - np.min(image_combined))
        
        # Convert to 8-bit image for PIL
image_combined = img_as_ubyte(image_combined)
        
        # Convert numpy array back to PIL Image
image = Image.fromarray(image_combined)


transform = transforms.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
])
image = transform(image)
transformed_image = image.unsqueeze(0)
with torch.no_grad():
    output = model(transformed_image)
    # Remove Comment For SWIN transformer
    #logits = output.logits 
    #replace output with logits for SWIN
    _, predicted_indices = torch.max(output, 1)
    predicted_index = predicted_indices.item()  # Convert tensor to Python number


predicted = round(predicted_index)
if predicted == 0:
        print('negative')
elif predicted == 1:
        print('postive')