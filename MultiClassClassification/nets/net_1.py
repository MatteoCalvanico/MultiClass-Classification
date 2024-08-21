import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from pytorch_model_summary import summary


# Rete di classificazione immagini.
class Net(nn.Module):

    def __init__(self, classes : list[str]) -> None:
        super(Net, self).__init__()

        # Strato convoluzionale, l'input ha 3 canali, l'output ne avra' 6.
        # Sara' analizzato da una "lente di ingrandimento" 5x5.
        self.conv1 = nn.Conv2d(3, 6, 5)

        # Strato di max pooling. Estrae il massimo valore concentrandosi,
        # posizione per posizione, su di una regione 2x2.
        self.pool = nn.MaxPool2d(2, 2)

        # Strato convoluzionale, l'input ha 6 canali, l'output ne avra' 16.
        # Sara' analizzato da una "lente di ingrandimento" 5x5.
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Primo strato di neuroni completamente connessi.
        # In input la totalita' dei neuroni precedenti, in output 120.
        self.fc1 = nn.Linear(16 * 53 * 53, 120)

        # Secondo strato di neuroni completamente connessi.
        # In input 120 neuroni, in output 84.
        self.fc2 = nn.Linear(120, 84)

        # Secondo strato di neuroni completamente connessi.
        # In input 84 neuroni, in output un neurone per classe.
        self.fc3 = nn.Linear(84, len(classes))

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))    # Convoluzione, ReLU, MaxPooling.
        x = self.pool(F.relu(self.conv2(x)))    # Convoluzione, ReLU, MaxPooling.
        x = torch.flatten(x, 1)                 # Tutti gli attuali neuroni vengono allineati in un unica 'fila'.
        x = F.relu(self.fc1(x))                 # Strato completamente connesso, ReLU.
        x = F.relu(self.fc2(x))                 # Strato completamente connesso, ReLU.
        x = self.fc3(x)                         # Strato completamente connesso.       
        return x                                # Output.
    
    def get_transforms(self):
        return transforms.Compose([transforms.ToTensor()])
    
if __name__ == '__main__':

    batch_size  = 1
    channels    = 3
    width       = 224
    height      = 224

    # Definisce la dimensione dell'input che avra' la rete.
    input_shape = (batch_size, channels, width, height)
    
    # Crea l'oggetto che rappresenta la rete.
    # Fornisce le classi.
    n = Net(['a', 'b', 'c'])
    
    # Salva i parametri addestrati della rete.
    torch.save(n.state_dict(), './out/model_state_dict.pth')
    
    # Salva l'intero modello.
    torch.save(n, './out/model.pth')
    
    # Stampa informazioni generali sul modello.
    print(n)

    # Stampa i parametri addestrabili.
    # for name, param in n.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

    # Stampa un recap del modello.
    print(summary(n, torch.ones(size=input_shape)))