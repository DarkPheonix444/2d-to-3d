from PIL import Image
import os


class ImageLoader:

    def __init__(self,file_path:str):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        self.file_path = file_path

    def load(self)->Image.Image:
        try:
            image = Image.open(self.file_path)

            img=image.convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {e}")
        
        return img
