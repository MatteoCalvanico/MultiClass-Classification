import cv2
import glob
import sys
import os
import matplotlib.pyplot as plt

from pathlib import Path
from visual_util import ColoredPrint as cp

class Balancer():
    
    def __init__(self, infoDir: dict, root : str, debug: bool = False) -> None:
        
        # Controlliamo se è necessario bilanciare. 
        # Per farlo vediamo se il dizionario con le informazioni è vuoto
        if not infoDir:  
            cp.red(f"No need to balance this root: {root}")
            sys.exit(-1)
        
        cp.purple('Initializing balancer...')
        
        self.data_path = Path(root)
        self.debug = debug
        
        # Per ogni classe facciamo delle trasformazioni
        for classes, img in infoDir.items():
            paths = glob.glob(f"{self.data_path}/{classes}_?") # Ci salviamo i percorsi completi delle directory da dove sono prese le immagini di ogni classe. Il "?" indica un solo valore dopo il "_" 
            
            self.__trasform(Path(paths[0]), img) # Creiamo tot immagini nuove nella prima directory trovata facendo trasformazioni sulle immagini giù presenti
            
    
    def __trasform(self, path: Path, numImg) -> None:

        while numImg > 0:
            for img in path.glob('**/*'):
                cv2Img = cv2.imread(img) # Leggiamo l'immagine con cv2
                
                cv2Img = self.__resize(cv2Img) # Facciamo un resize
                cv2Img = self.__rotate(cv2Img) # Facciamo una rotazione
                cv2Img = self.__translate(cv2Img) # Trasliamo l'immagine
                
                base_name, ext = os.path.splitext(img) #Otteniamo l'estenzione dell'immagine di partenza (e il nome)
                cv2.imwrite("balImg_" + str(numImg) + ext, cv2Img)
                
                numImg = numImg - 1
    
    def __resize(self, img):
        pass
    
    def __rotate(self, img):
        pass
    
    def __translate(self, img):
        pass
        

if __name__ == "__main__":
    
    b = Balancer({'prova': 1}, "../dataset/fruits-360_dataset_original-size/fruits-360-original-size/Test")