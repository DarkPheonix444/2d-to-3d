import cv2
import numpy as np
from PIL import Image
from typing import List


class normalizer:

    def __init__(self,target_width:int=1500):
        self.target_width = target_width

    def normalize(self,images:List[Image.Image])->List[np.ndarray]:

        processed_image=[]


        for img in images:
            
            img_np=np.array(img)

            img_np=cv2.cvtColor(img_np,cv2.COLOR_RGB2BGR)

            img_np=self.resize_if_needed(img_np)

            gray=cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (5, 5), 0)


            thresh = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )


            kernel=np.ones((3,3),np.uint8)

            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

            deskewed=self._deskew(morph)

            processed_image.append(deskewed)

        return processed_image
    

    def resize_if_needed(self,img:np.ndarray)->np.ndarray:

        h,w=img.shape[:2]

        if w>self.target_width:

            scale=self.target_width/w

            new_size=(self.target_width,int(h*scale)) 

            image=cv2.resize(img,new_size)

        return image


    def _deskew(self,img:np.ndarray)->np.ndarray:

        coords=np.column_stack(np.where(img>0))

        if len(coords)==0:
            return img


        angle=cv2.minAreaRect(coords)[-1]

        if angle<-45:
            angle=-(90+angle)

        else:
            angle=-angle


        (h,w)=img.shape[:2]

        center=(w//2,h//2)

        rotation_matrix=cv2.getRotationMatrix2D(center,angle,1.0)

        rotated = cv2.warpAffine(
            img,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated       