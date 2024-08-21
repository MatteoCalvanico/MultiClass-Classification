import sys
import numpy as np

np.random.seed(42)


class Metrics():
    
    def __init__(self, classes : list[str], real_y: np.array, pred_y: np.array) -> None:
        
        self.classes = classes
        self.num_classes = len(classes)
        
        self.real_y = real_y
        self.pred_y = pred_y
        
        self.confusion_matrix = None
    
    # Calcola la matrice di confusione fra le classi.
    def compute_confusion_matrix(self) -> None:
        
        # Crea una matrice di zeri grande N x N.
        # N rappresenta il numero delle classi.
        N = self.num_classes
        self.confusion_matrix = np.zeros((N, N), dtype=int)    
        
        # Aggiorna poi il conteggio fra classi reali e predette.
        for real, pred in zip(self.real_y, self.pred_y):
            self.confusion_matrix[real][pred] += 1
    
    # Calcola l'accuracy sulla confusion matrix.
    def accuracy(self) -> float:
        
        # Se la confusion matrix non e' ancora stata calcolata, lo fa.
        if self.confusion_matrix is None:
            self.compute_confusion_matrix()
        
        # Elementi sulla diagonale rispetto al totale.
        return np.sum(self.confusion_matrix.diagonal()) / np.sum(self.confusion_matrix)
    
    # Calcola la recall per la classe indicata.
    def recall(self, class_id: int) -> float:
        
        if not self.__valid_class_id(class_id):
            print(f'Id classe {class_id} invalido.')
            sys.exit(-1)
        
        # Se la confusion matrix non e' ancora stata calcolata, lo fa.
        if self.confusion_matrix is None:
            self.compute_confusion_matrix()
            
        # Il numero di veri positivi che sono anche stati predetti.
        real_and_predicted = self.confusion_matrix[class_id, class_id]
        
        # Il numero di veri positivi che si sarebbero dovuti predire.
        real = np.sum(self.confusion_matrix[class_id, :])           
        
        return 0.0 if real == 0 else (real_and_predicted / real)
    
    # Calcola la precision per la classe indicata.
    def precision(self, class_id: int) -> float:
        
        if not self.__valid_class_id(class_id):
            print(f'Id classe {class_id} invalido.')
            sys.exit(-1)
        
        # Se la confusion matrix non e' ancora stata calcolata, lo fa.
        if self.confusion_matrix is None:
            self.compute_confusion_matrix()
            
        # Il numero di veri positivi che sono anche stati predetti.
        real_and_predicted = self.confusion_matrix[class_id, class_id]
        
        # Il numero totale di positivi che sono stati predetti.
        predicted = np.sum(self.confusion_matrix[:, class_id])
        
        return 0.0 if predicted == 0 else (real_and_predicted / predicted)
    
    # Calcola l'f1-score per la classe indicata.
    def f1_score(self, class_id: int) -> float:
        
        # Calcola precision e recall per la classe.
        p, r = self.precision(class_id), self.recall(class_id)
        
        return 0.0 if (p + r) == 0 else 2 * (p * r) / (p + r)
    
    # Calcola il numero di campioni veri per la classe indicata.
    def support(self, class_id: int) -> int:
        
        if not self.__valid_class_id(class_id):
            print(f'Id classe {class_id} invalido.')
            sys.exit(-1)
        
        # Se la confusion matrix non e' ancora stata calcolata, lo fa.
        if self.confusion_matrix is None:
            self.compute_confusion_matrix()
            
        return np.sum(self.confusion_matrix[class_id, :])
    
    # Mostra un report di tutte le informazioni.
    def report(self) -> None:
        
        if self.confusion_matrix is None:
            self.compute_confusion_matrix()
        
        print('')
        print(f'Confusion matrix (accuracy {self.accuracy():.2f}%):\n\n{self.confusion_matrix}')        
        print(f'\nClass\t\tPrecision\tRecall\tf1-score\tsupport')
        for i, c in enumerate(self.classes):
            print(f'[{c}]\t{self.precision(i):.2f}\t\t{self.recall(i):.2f}\t{self.f1_score(i):.2f}\t\t{self.support(i):.2f}')
        print('')

    # Verifica se l'indice classe e' valido.
    def __valid_class_id(self, class_id: int) -> bool:
        return 0 <= class_id < len(self.classes)

if __name__ == '__main__':
    
    mt = Metrics(['circle', 'square', 'triangle'], 20)
    mt.report()

