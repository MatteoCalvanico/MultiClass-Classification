import sys
import re

from pathlib import Path
from torch.utils.data import Dataset
from visual_util import ColoredPrint as cp

class Analyzer(Dataset):
    
    def __init__(self, root : str, debug: bool = False) -> None:
        
        self.data_path = Path(root)
        self.debug = debug
        self.info = {} # Qui andranno tutti i nomi delle classi sbilanciate, con anche quante immagini in più dovrebbero avere rispetto alla xlasse/directory con più dati
        
        # Per prima cosa si controlla il percorso passato in 'root':
        # - Esiste?
        # - E' una cartella?
        # Se ci sono problemi, esco dallo script.
        if not self.__analyze_root():
            cp.red("Error: path do not exist or is a directory !!!")
            sys.exit(-1)
        
        # A questo punto la cartella e' valida:
        # - Controllo se ci sono file immagine all'interno.
        # - Le immagini le considero con estensione bmp, png, jpeg e jpg.
        # Se non ci sono immagini, esco dallo script.
        if not self.__search_image_files():
            cp.red("Error: the directory do not have any images !!!")
            sys.exit(-1)

        # A questo punto controllo la struttura di sotto-cartelle e file:
        # - Voglio che nella root ci sia un solo livello di sotto-cartelle.
        # - Il nome di ogni sotto-cartella sara' una classe di immagini.
        # - In ogni sotto-cartella ci saranno solo immagini di quella classe.  
        if not self.__check_structure():
            cp.red("Error: the root have more than one level !!!")
            sys.exit(-1)
        
        # Ora controllo se ogni sotto cartella è bilanciata
        self.__check_balance()

    def __analyze_root(self) -> bool:
        
        if self.debug:
            print(f'Analisi del percorso: {self.data_path.as_posix()}')
        
        if self.data_path.exists():
            if not self.data_path.is_dir():
                if self.debug:
                    print(f'{self.data_path.as_posix()} non e\' una cartella valida.')
                return False
        else:
            if self.debug:
                print(f'Cartella {self.data_path.as_posix()} inesistente.')
            return False

        if self.debug:
            print(f'Il percorso e\' valido.')

        return True
        
    def __search_image_files(self) -> bool:
        
        image_extensions = ('.bmp', '.png', '.jpeg', '.jpg')
        
        self.image_files = [x for x in self.data_path.glob('**/*') 
                            if x.is_file() and x.suffix in image_extensions]
        
        if len(self.image_files) > 0:
            if self.debug:
                print(f'Nella cartella sono stati trovati {len(self.image_files)} file immagine.')
            return True
        else:
            if self.debug:
                print(f'Nessuna immagine valida trovata.')
            return False

    def __check_structure(self) -> bool:
        
        # Perche' la struttura sia valida:
        # 1. Tutte le immagini devono avere una cartella padre e una cartella nonno (la root).
        # 2. La cartella nonno deve essere la root.
        condition_1 = all(len(f.parts) > 2 for f in self.image_files)
        condition_2 = all(f.parent.parent == self.data_path for f in self.image_files)
        
        if condition_1 and condition_2:
            if self.debug:
                print(f'La struttura delle sotto-cartelle in {self.data_path} e\' valida.')
            return True
        else:
            if self.debug:
                print(f'La struttura delle sotto-cartelle in {self.data_path} non e\' valida.')
            return False
        
    def __check_balance(self) -> None:
        
        counts = {} # Dizionario per contare le immagini per cartella
        
        for image_file in self.image_files:
            folder_name = image_file.parent.name
            base_name = re.sub(r'_(\d+)', '', folder_name)
            counts[base_name] = counts.get(base_name, 0) + 1

        # Trovo il numero massimo di immagini
        max_count = max(counts.values())
        
        #Mi salvo quali classi hanno bisogno di essere bilanciate, e di quanto
        for folder, count in counts.items():
            if count < max_count:
                self.info[folder] = max_count - count


if __name__ == '__main__':

    cds = Analyzer('../dataset/fruits-360_dataset_original-size/fruits-360-original-size/Training')
    
    if not cds.info:
        print("Nessuna cartella sbilanciata.")
        sys.exit(-1)
    
    for i, data in cds.info.items():
        print(f'Dir: {i} ha bisogno di [{data}] nuove immagini per essere bilanciata')