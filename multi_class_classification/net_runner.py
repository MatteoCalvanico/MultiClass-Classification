import os
import sys
import math
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import importlib

from pathlib import Path
from metrics import Metrics
from visual_util import ColoredPrint as cp
from custom_dataset_fruits import CustomDataset

torch.manual_seed(42)
np.random.seed(42)


class NetRunner():

    def __init__(self, cfg_object: object, classes: list[str]) -> None:
        
        cp.purple('Initializing net runner...')
        
        # Salvo il file di configurazione.
        self.cfg = cfg_object
        
        # Salvo le classi del dataset.
        self.classes = classes
        cp.cyan(f'Classifier classes: {self.classes}')
        
        # Acquisisco la rete, in base al tipo richiesto.
        self.net = self.__get_net()
        
        # Carico e predispongo i loader dei dataset.
        self.__load_data()

        # Predispone la cartella di output.
        self.out_root = Path(self.cfg.io.out_folder)
        
        # Il percorso indicato esiste?
        if not self.out_root.exists():
            cp.cyan(f'Creating output directory: {self.out_root}')
            self.out_root.mkdir()
        
        # Indico dove salvero' il modello addestrato.
        self.last_model_outpath_sd = self.out_root / 'last_model_sd.pth'
        self.last_model_outpath = self.out_root / 'last_model.pth'
        self.best_model_outpath_sd = self.out_root / 'best_model_sd.pth'
        self.best_model_outpath = self.out_root / 'best_model.pth'
        
        # Se richiesto, si cerca l'ultimo modello salvato in fase di addestramento.
        # Di lui ci interessa lo stato dei pesi, lo state_dict.
        # Se presente, la rete sara' inizializzata a quello stato.
        if self.cfg.train_parameters.reload_last_model:
            try:
                self.net.load_state_dict(torch.load(self.last_model_outpath_sd))
                cp.green('Last model state_dict successfully reloaded.')
            except:
                cp.red('Cannot reload last model state_dict.')
        
        # Funzione di costo.
        cp.cyan(f'Created loss function.')
        self.criterion = nn.CrossEntropyLoss()
        
        # Ottimizzatore.
        cp.cyan(f'Created optimizer (lr: {self.cfg.hyper_parameters.learning_rate}, m: {self.cfg.hyper_parameters.momentum}).')
        self.optimizer = optim.SGD(self.net.parameters(), 
                                   lr = self.cfg.hyper_parameters.learning_rate, 
                                   momentum = self.cfg.hyper_parameters.momentum)        

    def train(self) -> None:
        
        cp.purple("Training...")
        
        # Conteggio degli step totali.
        global_step = 0
        
        epochs = self.cfg.hyper_parameters.epochs
        cp.cyan(f'Training loop epochs: {epochs}')

        # Salvo in una variabile in modo da mostrare una sola volta.
        show_preview = self.cfg.parameters.show_preview
        
        # Ogni quanto monitorare la funzione di costo.
        train_step_monitor = self.cfg.train_parameters.step_monitor
        # Ogni quante epoche valutare l'accuracy.
        accuracy_evaluation_epochs = self.cfg.train_parameters.accuracy_evaluation_epochs
        # Il target di accuracy da raggiungere.
        accuracy_target = self.cfg.train_parameters.accuracy_target
        
        cp.cyan(f'Training monitor every {train_step_monitor} steps, preview: {show_preview}.')
        cp.cyan(f'Training and validation accuracy evaluated every {accuracy_evaluation_epochs} epochs.')
        cp.cyan(f'Accuracy target to reach is {accuracy_target}%.')
        
        es_start_epoch = self.cfg.early_stop_parameters.start_epoch
        es_loss_evaluation_epochs = self.cfg.early_stop_parameters.loss_evaluation_epochs
        es_patience = self.cfg.early_stop_parameters.patience
        es_improvement_rate = self.cfg.early_stop_parameters.improvement_rate
        
        cp.cyan(f'Early stop check will start at epoch {es_start_epoch}.')
        cp.cyan(f'Validation loss evaluated every {es_loss_evaluation_epochs} epochs.')
        cp.cyan(f'Early stop will be triggered after {es_patience} epochs of no improvement.')
        cp.cyan(f'Minimum requested improvement is {es_improvement_rate}% on validation loss.')

        tr_losses_x, tr_losses_y = [], []           # Raccoglitori loss di training.
        tr_run_losses_x, tr_run_losses_y = [], []   # Raccoglitori loss di training ogni step del monitor.
        va_losses_x, va_losses_y = [], []           # Raccoglitori loss di validazione.
        
        best_tr_acc = float('-inf') # Traccia la migliore accuracy raggiunta in addestramento.
        best_va_acc = float('-inf') # Traccia la migliore accuracy raggiunta in validazione.
        best_va_loss = float('inf') # Traccia la migliore loss raggiunta in validazione.
        
        # Con questo contatore, si valuta per quanti check consecutivi
        # la loss di validazione non e' migliorata.
        va_loss_no_improve_ep_ctr = 0
        
        target_accuracy_reached = False # FLAG EVENTO: il training puo' fermarsi per accuratezza target raggiunta.
        early_stop_check = False        # FLAG EVENTO: puo' iniziare il check regolare per l'early stop.
        early_stop = False              # FLAG EVENTO: l'early stop e' scattato, stop dell'addestramento.

        # Loop di addestramento per n epoche.
        for epoch in range(epochs):
            
            # ********************
            #if (epoch + 1) == 10:
            #    break
            # ********************
            
            # L'analisi dell'early stop inizia solo quando sono passate le epoche richieste.
            if (epoch + 1) == es_start_epoch:
                early_stop_check = True
            
            # Se l'early stop e' scattato, ci si ferma.
            if early_stop:
                cp.yellow('Stopping: detected EarlyStop!')
                break
            
            # Target performance raggiunta, stop dell'addestramento.
            if target_accuracy_reached:
                cp.green('Stopping: accuracy target reached for training and validation!')
                break

            running_loss = 0.0

            # Stop di addestramento. Dimensione batch_size.
            for i, data in enumerate(self.tr_loader, 0):

                # Le rete entra in modalita' addestramento.
                self.net.train()

                # Per ogni input tiene conto della sua etichetta.
                inputs, labels = data
                
                if show_preview:
                    self.__show_preview(inputs, labels)
                    show_preview = False

                # L'input attraversa al rete. Errori vengono commessi.
                # L'input diventa l'output.
                outputs = self.net(inputs)

                # Calcolo della funzione di costo sulla base di predizioni e previsioni.
                loss = self.criterion(outputs, labels)
                
                # I gradienti vengono azzerati.
                self.optimizer.zero_grad()

                # Avviene il passaggio inverso.
                loss.backward()
                
                # Passo di ottimizzazione
                self.optimizer.step()

                # Monitoraggio statistiche.
                running_loss += loss.item()
                
                if (i + 1) % train_step_monitor == 0:
                    tr_run_losses_y.append(running_loss / train_step_monitor)
                    tr_run_losses_x.append(global_step)
                    print(f'global_step: {global_step:5d} - [ep: {epoch + 1:3d}, step: {i + 1:5d}] loss: {loss.item():.6f} - running_loss: {(running_loss / train_step_monitor):.6f}')
                    running_loss = 0.0
                
                tr_losses_y.append(loss.item())
                tr_losses_x.append(global_step)

                global_step += 1
            
            # Controllo della loss di validazione:
            # - Se il check dell'early stop e' abilitato.
            # - E sono passate le epoche di attesa fra un check e l'altro.     
            if early_stop_check and (epoch + 1) % es_loss_evaluation_epochs == 0:
                
                cp.cyan("... Evaluating validation loss ...")
                
                current_va_loss = 0
                current_va_loss_counter = 0
                
                # Stop di validazione.
                for i, data in enumerate(self.va_loader, 0):
                    
                    # Le rete entra in modalita' addestramento.
                    self.net.eval()
                    
                    # Per ogni input tiene conto della sua etichetta.
                    inputs, labels = data

                    # Disabilita computazione dei gradienti.
                    with torch.no_grad():

                        # Esegue le predizioni.
                        outputs = self.net(inputs)

                        # Calcola la loss.
                        loss += self.criterion(outputs, labels)

                        current_va_loss += loss.item()
                        current_va_loss_counter += 1
                
                va_loss = current_va_loss / current_va_loss_counter
                va_losses_x.append(global_step)
                va_losses_y.append(va_loss)
                
                # Verifica se lo loss di validazione e' migliorata:
                # - Se non lo e', aggiurno il counter dei NON MIGLIORAMENTI.
                # - Se lo e' ma non a sufficienza, aggiurno il counter dei NON MIGLIORAMENTI.
                # - Se lo e', azzero il counter.
                if va_loss < best_va_loss:
                    
                    # Calcolo il tasso di miglioramento.
                    improve_ratio = abs((va_loss / best_va_loss) - 1) * 100
                    
                    # Verifico che il miglioramento non sia inferiore al tasso.
                    if improve_ratio >= es_improvement_rate:
                        cp.green(f'... validation loss improved: {best_va_loss:.6f} --> {va_loss:.6f} ({improve_ratio:.3f}%)')
                        best_va_loss = va_loss
                        va_loss_no_improve_ep_ctr = 0
                    else:
                        cp.red(f'... validation loss NOT improved ... ({improve_ratio:.3f}%) < ({es_improvement_rate}%)')
                        va_loss_no_improve_ep_ctr += 1
                else:
                    cp.red(f'... validation loss NOT improved')
                    va_loss_no_improve_ep_ctr += 1
            
            # Calcolo l'accuratezza sui dati di training e validazione:
            # - Se sono passate il numero atteso di epoche.
            # - Se e' l'ultima epoca di training.
            if (epoch + 1) % accuracy_evaluation_epochs == 0 or (epoch + 1) == epochs:
                
                cp.cyan("... Evaluating accuracy for training and validation data...")
                
                # Valutazione su datset di addestramento.
                tr_acc = self.test(self.tr_loader, use_current_net=True) * 100
                cp.blue('...on training data...')
                
                # Valutazione su dataset di validazione.
                va_acc = self.test(self.va_loader, use_current_net=True) * 100
                cp.blue('...on validation data...')
                
                tr_improved, va_improved = False, False
                
                # L'accuracy di training e' migliorata?
                if tr_acc > best_tr_acc:
                    best_tr_acc = tr_acc
                    tr_improved = True
                
                # L'accuracy di validazione e' migliorata.
                if va_acc > best_va_acc:
                    best_va_acc = va_acc
                    va_improved = True
                
                # Se sono entrambe migliorate, lo considero un buon momento per
                # salvare lo stato di questi pesi e considerarli il nuovo MODELLO
                # MIGLIORE.
                if tr_improved and va_improved:
                    torch.save(self.net.state_dict(), self.best_model_outpath_sd)
                    torch.save(self.net, self.best_model_outpath)
                    cp.green('Best model saved.')
                
                cp.yellow(f'Training accuracy: {(tr_acc):.2f}% - Validation accuracy: {(va_acc):.2f}%')
                
                # Se entrambe le accuracy hanno raggiunto il target, alzo il FLAG.
                # Prima della prossima epoca, l'addestramento si fermera'.
                if best_tr_acc > accuracy_target and best_va_acc > accuracy_target:
                    target_accuracy_reached = True                

            # Se la loss di validazione non migliora da 'patience' epoche, e' il
            # momento di alzare il FLAG e richiedere l'early stop.
            if va_loss_no_improve_ep_ctr >= es_patience:
                early_stop = True

        # Salvo l'ultimo stato/modello a termine dell'addestramento.
        torch.save(self.net.state_dict(), self.last_model_outpath_sd)
        torch.save(self.net, self.last_model_outpath)
        cp.yellow('Last model saved.')
        
        cp.blue('Finished Training.')

        _, (ax1, ax2) = plt.subplots(2, 1)
        
        ax1.plot(tr_losses_x, tr_losses_y, label='train_loss')
        ax1.plot(tr_run_losses_x, tr_run_losses_y, label='train_running_loss')
        ax1.set_title('Training loss')
        ax1.legend()
        
        ax2.plot(va_losses_x, va_losses_y, label='validation_loss')
        ax2.set_title('Validation loss')
        ax2.legend()

        plt.tight_layout()
        plt.show()
    
    def test(self, loader: torch.utils.data.DataLoader = None, use_current_net: bool = False, preview : bool = False, print_acc: bool = False):

        cp.purple("Testing...")

        # Se non specifico diversamente, testo sui dati di test.
        if loader is None:
            loader = self.te_loader

        # Se richiesto, testo sul modello corrente e non il migliore.
        if use_current_net:
            net = self.net
        else:
            # Per usare il modello migliore:
            # - Richiedo un modello 'nuovo'.
            # - Inizializzo i suoi pesi allo stato del modello migliore.
            net = self.__get_net()
            
            try:
                net.load_state_dict(torch.load(self.best_model_outpath_sd))
            except:
                print('Missing model state_dict.')
                return

        # La rete entra in modalita' inferenza.
        net.eval()
        
        real_y = []
        pred_y = []

        # Non e' necessario calcolare i gradienti al passaggio dei dati in rete.
        with torch.no_grad():

            # Cicla i campioni di test, batch per volta.
            for data in loader:

                # Dal batch si estraggono dati ed etichette.
                images, labels = data
                
                if preview:
                    self.__show_preview(images, labels)

                # I dati passano nella rete e generano gli output.
                outputs = net(images)

                # Dagli output si evince la predizione finale ottenuta.
                _, predicted = torch.max(outputs.data, 1)
                
                real_y = real_y + labels.tolist()
                pred_y = pred_y + predicted.tolist()
            
        # Utilizzo metrics per calcolare le statistiche sui dati.
        mt = Metrics(self.classes, real_y, pred_y)
        
        if print_acc:
            cp.yellow(f'Test accuracy: {mt.accuracy():.2f}%')
    
        return mt.accuracy()
    
    def denormalize_v1(self, img):
        return np.transpose(img.numpy(), (1, 2, 0))

    # Mostra una preview di immagini ed etichette in una griglia.
    def __show_preview(self, images, labels):

        cols = 8
        rows = math.ceil(len(images) / cols)

        _, axs = plt.subplots(rows, cols, figsize=(18, 9))
        axs = axs.reshape(rows * cols)
        for ax, im, _ in zip(axs, images, labels):
            ax.imshow(self.denormalize_v1(im))
            #ax.set_title(self.classes[lb.item()])
            ax.grid(False)

        plt.show()
        
    # Ottiene un oggetto 'rete' del tipo richiesto.
    def __get_net(self):
        
        selectedNet = self.cfg.train_parameters.network_type.lower()
        netsPath = os.path.join(self.cfg.io.nets_folder.lower(), f"{selectedNet}.py")
        
        if not os.path.exists(netsPath):
            cp.red(f'Unknown net.')
            sys.exit(-1)
        
        # Importiamo il modulo dinamicamente
        spec = importlib.util.spec_from_file_location(selectedNet, netsPath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Prendiamo la classe dal modulo appena importato
        Net = getattr(module, 'Net')
        
        
        #if self.cfg.train_parameters.network_type.lower() == 'net_1':
        #    from nets.net_1 import Net
        #elif self.cfg.train_parameters.network_type.lower() == 'net_2':
        #    from nets.net import Net
        #else:
        #    print(f'Unknown net.')
        #    sys.exit(-1)
        
            
        return Net(self.classes)
    
    # Carica i Dataset tramite Dataloader e scopre le classi del dataset.
    def __load_data(self) -> None:
    
        transforms = self.net.get_transforms()
    
        # Posso raccogliere le immagini con il dataset custom creato appositamente.
        # - Posso farlo sia per i dati di training.
        # - Che per quelli di test e/o validazione.
        cp.cyan(f'Analyzing training dataset: {self.cfg.io.training_folder}')
        tr_dataset = CustomDataset(root=self.cfg.io.training_folder, transform=transforms)
        cp.cyan(f'Analyzing validation dataset: {self.cfg.io.validation_folder}')
        va_dataset = CustomDataset(root=self.cfg.io.validation_folder, transform=transforms)
        cp.cyan(f'Analyzing test dataset: {self.cfg.io.test_folder}')
        te_dataset = CustomDataset(root=self.cfg.io.test_folder, transform=transforms)
        self.classes = tr_dataset.classes

        # Se non voglio usare il dataset custom, posso usarne uno di base fornito da PyTorch.
        # Questo rappresenta genericamente:
        # - Un dataset di immagini.
        # - Diviso in sotto-cartelle.
        # - Il nome delle sotto-cartelle rappresenta il nome della classe.
        # - In ogni sotto-cartella ci sono solo immagini di quella classe.
        if not self.cfg.io.use_custom_dataset:
            tr_dataset = torchvision.datasets.ImageFolder(root=self.cfg.io.training_folder, transform=transforms)
            va_dataset = torchvision.datasets.ImageFolder(root=self.cfg.io.validation_folder, transform=transforms)
            te_dataset = torchvision.datasets.ImageFolder(root=self.cfg.io.test_folder, transform=transforms)

        # In entrambi i casi passo le trasformazioni da applicare.

        # Creo poi i dataloader che prendono i dati dal dataset:
        # - lo fanno a pezzi di dimensione 'use_custom_dataset'.
        # - i pezzi li compongono di campioni rando se abilitato 'shuffle'.
        self.tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=self.cfg.hyper_parameters.batch_size, shuffle=True)
        self.va_loader = torch.utils.data.DataLoader(va_dataset, batch_size=self.cfg.hyper_parameters.batch_size, shuffle=False)
        self.te_loader = torch.utils.data.DataLoader(te_dataset, batch_size=self.cfg.hyper_parameters.batch_size, shuffle=False)   