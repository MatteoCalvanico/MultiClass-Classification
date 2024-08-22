import sys
import cv2
import re

from pathlib import Path
from torch.utils.data import Dataset
from visual_util import ColoredPrint as cp


class CustomDataset(Dataset):

    def __init__(self, root : str, transform = None, debug: bool = False) -> None:

        self.debug = debug
        
        # Memorizzo le trasformazioni che potro' fare sui dati.
        self.transform = transform
                
        self.data_path = Path(root)
        
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
        
        # Con la certezza che la struttura delle cartelle e' corretta, si
        # possono estrarre i nomi delle classi dai nomi delle sotto-cartelle.
        # Ad ogni classe, orinata in senso alfabetico, si da un indice.
        self.__find_classes_and_labels()
        
        # Trovate classi e indici, si assegna ad ogni immagine una etichetta.
        self.__assign_labels()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        
        img = cv2.imread(self.image_files[index].as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        class_idx = self.labels[index]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx

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

    def __find_classes_and_labels(self) -> None:
        
        # Raccolgo i nomi di tutte le sotto-cartelle, anche duplicati.
        sub_folder_names = [f.parts[-2] for f in self.image_files]
        
        # Rimuovo i numeri dalle classi e creo un mapping tra i nomi vecchi (con i numeri) e i nuovi
        self.original_to_cleaned = {}
        for name in sub_folder_names:
            cleaned_name = re.sub(r'\d+', '', name)
            self.original_to_cleaned[name] = cleaned_name
        
        # Lascio solo valori univoci, senza numeri e ordinati alfabeticamente.
        self.classes = sorted(list(set(self.original_to_cleaned.values())))
        
        # Assegno ad ogni classe un indice
        self.class_labels = {c: i for i, c in enumerate(self.classes)}
        
        if self.debug:
            print('Classi trovate ed etichette assegnate:')
            for c, l in self.class_labels.items():
                if self.debug:
                    print(f'|__Classe [{c}]\t: etichetta [{l}]')

    def __assign_labels(self) -> None:
        
        self.labels = []
        
        class_distributions = {c: 0 for c in self.classes}
        
        for i in self.image_files:    
            original_class = i.parts[-2]
            cleaned_class = self.original_to_cleaned[original_class]
            class_distributions[cleaned_class] += 1
            self.labels.append(self.class_labels[cleaned_class])
        
        if self.debug:
            print('Distribuzione classi:')
            for c, d in class_distributions.items():
                if self.debug:
                    print(f'|__Classe [{c}]\t: {d} ({d/float(len(self.image_files)):.2f}%)')

if __name__ == '__main__':

    cds = CustomDataset('../dataset/fruits-360_dataset_original-size/fruits-360-original-size/Training')
    
    for i, data in enumerate(cds):
        if i == 10:
            break
        print(f'Campione {i}, etichetta: [{data[1]}]')
        cv2.imshow("Immagine", data[0])
        cv2.waitKey()
    
    cv2.destroyAllWindows()