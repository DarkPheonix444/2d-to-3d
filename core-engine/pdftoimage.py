from PIL import Image

from pdf2image import convert_from_path
import os
from typing import List,Optional

class PDFToImage:

    def __init__(self, pdf_path:str,dpi:int=300,max_pages:int=50):
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.pdf_path = pdf_path
        self.dpi = dpi
        self.max_pages = max_pages
        self.pages: List[Image.Image] = []

    def convert_to_image(self,first_page:Optional[int]=None,last_page:Optional[int]=None)->List[Image.Image]:
        if first_page is None:
            first_page = 1
        if last_page is None:
            last_page = self.max_pages
        else:
            last_page = min(last_page,self.max_pages)
        
        try:
            self.pages=convert_from_path(
                self.pdf_path,
                dpi=self.dpi,
                first_page=first_page,
                last_page=last_page
            )
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {e}")
        
        return self.pages
    
    def get_page(self,index:int)-> Image.Image:
        if not self.pages:
            raise ValueError("No pages have been converted yet. Please call convert_to_image() first.")
        if index < 0 or index >= len(self.pages):
            raise IndexError("page doesn't exist")
        
        
        return self.pages[index]



    def page_count(self)->int:
        return len(self.pages)

