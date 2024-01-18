"""
This script is for quick testing different image processing and IO methods
"""
from PIL import Image
import numpy as np
from torchvision import transforms as T
if __name__ == '__main__':
    """
    Load an mnist jpg image , normalize it , inverse normalize then re-save it 
    """
    sample_image_file = "../mnist_image_samples/0/img_1.jpg"
    img = Image.open(sample_image_file)
    raw_img_data = np.array(img)
    img_size = 32
    transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor()
    ])
    transformed_img = transform(img)
    reverse_img = (transformed_img * 255).squeeze().detach().numpy()
    out_image = Image.fromarray(reverse_img).convert("L")
    out_image.save("reversed.png")
    print("finished")
"""
Resources
https://stackoverflow.com/a/64629989 
"""
