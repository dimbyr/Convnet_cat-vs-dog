import os # os.walk: walking through directories
import numpy as np 
from torchvision import transforms
from PIL import Image

class load():
    """
    custom dataset class, creating an image dataset.
    """
    def __init__(self, img_dir = '', img_ext = '.jpg'):
        """
        img_dir : path pointing to the directory containing the images. 
        img_ext : extension of the images. It requires
                all of the images to have exactly the same
                extension. (e.g.: jpg, png, jpeg, ...)
        """
        self.img_ext = img_ext
        self.img_dir = img_dir
        self.transform = transforms.Compose([
             transforms.ToTensor(),
            transforms.Normalize((0.,), (1,))
             ])
    
    
    def image_names(self):
        '''
        return a list of all of the image names.
        '''
        names = []# list(os.walk(self.img_dir))
        for _,__,f in os.walk(self.img_dir):
            for file in f:
                if (file[-len(self.img_ext):] == self.img_ext) and ('check' not in file):
                    names.append(file)
        return names
    
    
    def get_category(self, categories=['cat', 'dog']):
        
        '''
        input:
            categories: list of categories.
            The images are assumed to be named accordingly to their
            categories
        output: list of the labels 
        '''
        
        labels = []
        for name in self.image_names():
            for c in categories:     
                if c.lower() in name.lower():
                    labels.append(c)
        return np.array(labels)
    
    
    def image_set(self, input_size = (28,28), as_gray = True):
        """
        Reads all of the images in the directory containing them,
        and put the matrices in a list.
        
        as_gray: Transform images to gray scale.
        input_size: Size of the resized images.
        """
        names = self.image_names()
        ims = []           
        for name in names:
            img = Image.open(self.img_dir + name).resize(input_size)
            if as_gray:
                img = img.convert('L')
            img = self.transform(img)
            ims.append(img)
        return ims

    def data(self, categories = ['cat', 'dog']):
        return zip(self.image_set(), list((self.get_category(categories)=='dog')*1.))
                           