# moltean/fruits MultiClass Classification

## Start-up
Per iniziare creare un nuovo enviroment di conda utilizzando i file all'interno della cartella *requirements*.

Successivamente far partire lo script *datasetDownloader.py* per scaricare e unzippare, tramite le API Kaggle, i dati che verranno utilizzati dalla rete.

Infine entrare nella cartella *multi_class_classification* e avviare *image_classifier.py*


## Context
Lo scopo ultimo della rete è di riuscire a classificare diversi tipi di frutta e dire a quale classe appartengono, le classi sono:
- Apple, con le sotto classi:
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

## Structure
La rete e i file di configurazione/utilità sono all'interno della cartella *multi_class_classification*, di seguito la spiegazione di ciascun file o directory che sono stati modificati rispetto al template di partenza:

### custom_dataset_fruits.py
FIle con il compito di estrarre le classi e dare le label alle immagini del dataset.

Cambiamenti fatti:
- in *__getitem__*: aggiunta la conversione in PIL Image;
- in *__find_classes_and_labels*: visto la struttura del dataset ora la funzione rimuove anche i numeri evitando duplicati che prima non venviano individuati;

### net_runner.py
FIle con il compito di inizializzare la rete, allenarla e controllarla.

Cambiamenti fatti:
- in *__get_net*: ora la rete viene presa dinamicamente in base a ciò che è inserito nel file di configurazione.