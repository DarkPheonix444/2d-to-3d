import os 
import mimetypes
from typing import List
from PIL import Image

from core.pdftoimage import PDFToImage
from core.imageloader import ImageLoader


supported_image_types=[
    'image/jpeg',
    'image/png',
    'image/jpg'
]


class InputController:



    def process(self,file_path:str)->list[Image.Image]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.file_path = file_path
        mime_types,_=mimetypes.guess_type(self.file_path)

        if mime_types in supported_image_types:
            return self.handle_image()
        elif mime_types=='application/pdf':
            return self.handle_pdf()
        
        raise ValueError(f"Unsupported file type: {mime_types}")
    

    def handle_image(self)->List[Image.Image]:
        loader=ImageLoader(self.file_path)
        image=loader.load()
        return [image]
    

    def handle_pdf(self)->List[Image.Image]:
        pdf_processor=PDFToImage(self.file_path)
        images=pdf_processor.convert_to_image()
        return images