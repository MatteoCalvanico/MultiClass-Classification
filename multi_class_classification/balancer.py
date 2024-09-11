import cv2
import glob
import sys
import os
import random
import numpy as np

from pathlib import Path
from visual_util import ColoredPrint as cp

class Balancer():
    
    def __init__(self, infoDir: dict, root : str, debug: bool = False) -> None:
        
        # Controlliamo se è necessario bilanciare. 
        # Per farlo vediamo se il dizionario con le informazioni è vuoto
        if not infoDir:  
            cp.red(f"No need to balance this root: {root}")
            sys.exit(-1)
        
        cp.purple(f'Initializing balancer in {root}...')
        
        self.data_path = Path(root)
        self.debug = debug
        
        # Per ogni classe facciamo delle trasformazioni
        for classes, img in infoDir.items():
            cp.yellow(f"Starting to balancing: {classes} class")
            
            paths = glob.glob(f"{self.data_path}/{classes}_?") # Ci salviamo i percorsi completi delle directory da dove sono prese le immagini di ogni classe. Il "?" indica un solo valore dopo il "_" 
            
            self.__trasform(Path(paths[0]), img) # Creiamo tot immagini nuove nella prima directory trovata facendo trasformazioni sulle immagini giù presenti
            
    
    def __trasform(self, path: Path, numImg) -> None:

        for img in path.glob('**/*'):
            
            if numImg <= 0: #Quando abbiamo abbastanza immagini ci fermiamo
                break
            
            cv2Img = cv2.imread(img) # Leggiamo l'immagine con cv2
                
            cv2Img = self.__rotate(cv2Img)    # Facciamo una rotazione
            cv2Img = self.__translate(cv2Img) # Trasliamo l'immagine
                
            base_name, ext = os.path.splitext(img) #Otteniamo l'estenzione dell'immagine di partenza (e il nome)
            cv2.imwrite(os.path.join(path, "balImg_" + str(numImg) + ext), cv2Img)
                
            numImg -= 1
    
    def __rotate(self, img):
        rows, cols = img.shape[0], img.shape[1]
        
        image_center = (cols / 2, rows / 2)                 # Punto di rotazione al centro immagine.
        custom_rotation_angle = random.uniform(0.0, 360.0)  # Angolo di rotazione personalizzato.
        scale = 1.0                                         # Fattore di scala applicabile all'immagine.
        
        transformation_matrix = cv2.getRotationMatrix2D(image_center, custom_rotation_angle, scale) # Costruisco una matrice di rotazione da applicare all'immagine.
        rotated_img = cv2.warpAffine(img, transformation_matrix, (cols, rows))    # Applico la rotazione all'immagine
        
        return rotated_img
    
    def __translate(self, img):
        rows, cols = img.shape[0], img.shape[1]
        
        transformation_matrix = np.float32([[1,0,50],[0,1,25]])     # Eseguo la traslazione in x di 50 pixel.
                                                                    # Eseguo la traslazione in y di 25 pixel.                                                              
        translated_image = cv2.warpAffine(img, transformation_matrix, (cols, rows)) # Applico la trasformazione all'immagine.
        
        return translated_image
        

if __name__ == "__main__":
    
    b = Balancer({'prova': 1}, "../dataset/fruits-360_dataset_original-size/fruits-360-original-size/Test")