# moltean/fruits MultiClass Classification

## Start-up
Per iniziare creare un nuovo enviroment di conda utilizzando i file all'interno della cartella *requirements*.

Successivamente far partire lo script *datasetDownloader.py* per scaricare e unzippare, tramite le API Kaggle, i dati che verranno utilizzati dalla rete.

Infine entrare nella cartella *multi_class_classification* e avviare *image_classifier.py*.


## Context
Lo scopo ultimo della rete è di riuscire a classificare diversi tipi di frutta e dire a quale classe appartengono, le classi sono:
- Apple:
    - Braeburn;
    - Crimson;
    - Golden;
    - Granny;
    - Hit;
    - Pink Lady;
    - Red;
    - ...

- Cabbage:

- Carot;

- Cucumber;

- Eggplant;

- Pear;

- Zucchini.

## Architecture
- Tipologia di rete: Classic CNN
- Funzione di attivatione: Non lineare/ReLU (Rectified Linear Unit);
- Funzione di loss: Cross-Entropy;
- Ottimizzatore: SGD (Stochastic Gradient Descent).

La scelta della loss function di tipo *Cross-Entropy* è dovuta al fatto che è particolarmente adatta per problemi di classificazione multiclasse, penalizzando fortemente le previsioni errate con alta confidenza e integrando già Softmax.
Per l'ottimizzatore si è deciso di usare SDG perchè efficiente e robusto nel trovare buone soluzioni.

## Analysis & Experiments
Oltre alla rete e i suoi file potete trovare un file Notebook chiamato *Dataset_Analysis.ipynb* che mostra alcune informazioni sul dataset utilizzato.

Potete anche consultare i vari risultati ottenuti da diversi esperimenti come:
- addestramento per diversi numeri di epoche.
- addestramento con un diverso numero di layer.
- addestramento con diversi valori di learning rate.
  
Tutto nel file *Report.md*

## Tensorboard
Utilizzando, all'interno della directory contenente i vari file, il seguente comando:
```sh
tensorboard --logdir=out/runs --port=6006 --reload_interval=1
```
sarà possibile visualizzare:
- l'andamento, durante il train, della *loss* per ciascun training step e la *average loss* su più step;
- le **confusion matrix** di train e validation.

## Structure
```
root/
├── multi_class_classification/
│   ├── config/
│   │   ├── config.json             # File di configurazione che permette di modificare vari parametri del modello o di utilità
│   │   ├── config_schema.json      # Rappresenta come dovrebbe essere rappresentato il file: config.json
│   ├── nets/
│   │   ├── net.py                  # Base CNN
│   │   ├── net_deep.py             # Deep CNN
│   │   ├── net_wide.py             # Wide CNN
│   ├── out/
│   │   ├── ...                     # Modelli salvati e file per Tensorboard
│   ├── analyzer.py                 # Controlla quali classi sono sbilanciate
│   ├── balancer.py                 # Aggiunge immagini per compensare lo sblilanciamento delle classi trovate dall'analyzer.py
│   ├── config_helper.py            # Controlla, grazie a config_schema.json, se il file di configurazione è ben impostato
│   ├── custom_dataset_fruits.py    # Estrae le classi e assegna le label
│   ├── image_classifier.py         # File principale, serve per far partire l'addestramento e tutte le altri classi utils
│   ├── metrics.py                  # Calcola matrici di confusione, accuracy associata e altro ancora
│   ├── net_runner.py               # Inizializza la rete, la allena e la controlla
│   ├── visual_util.py              # Permette di scrivere in maniera più carina sulla console
├── requirements/
│   ├── ...                         # Files per il setup dell'ambiente CONDA
├── Dataset_Analysis.ipynb          # Analisi sul dataset
├── Report.md                       # Risultati ottenuti su vari test
├── ...
```
