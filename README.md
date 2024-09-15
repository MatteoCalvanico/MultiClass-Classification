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

## Structure
La rete e i file di configurazione/utilità sono all'interno della cartella *multi_class_classification*, di seguito la spiegazione dei file o directory più importanti:

### config directory
In questa cartella sono presenti:
- config.json: file di configurazione che permette di modificare vari parametri del modello o di utilità;
- config_schema.json: tramite questo file andremo a controllare che i valori inseriti nel file precedente siano corretti.

### nets directory
Qui vengono inserite le reti che vogliamo usare, basta inserire il nome nel file di configurazione per scegliere quella da addestrare; poi verrà presa dinamicamente dal *net_runner.py*.

### out directory
Qui vengono salvati i modelli addestrati tramite *image_classifier.py*, possiamo trovare:
- Modello migliore;
- Ultimo modello;

### image_classifier.py
File principale, si occupa di controllare/caricare il file di config, grazie a *config_helper.py*, e far partire le varie classi spiegate successivamente; in base alla configurazione in *config.json* decide se far partire l'analyzer e il balancer.

### analyzer.py
File con il compito di controllare quali classi sono sbilanciate, andato a controllare se per ogni cartella ci sono meno file rispetto a quelli della classe più grande.

Stampa anche la distribuzione delle classi e quante immagini ci sono al momento per classe.

### balancer.py
Prende dall'analyzer quali classi sono sbilanciate e aggiunge tante immagini quante ne sono richieste per bilanciare il tutto. Per farlo applica delle trasformazioni randomiche alle varie immagini (come rotazioni e traslazioni).

### custom_dataset_fruits.py
FIle con il compito di estrarre le classi e dare le label alle immagini del dataset.

Nel dettaglio:
- tramite il parametro *skip* si evita il controllo delle directory se fatto già dall'analyzer;
- in *\__getitem__*: aggiunta la conversione in PIL Image;
- in *__find_classes_and_labels*: visto la struttura del dataset la funzione rimuove anche i numeri evitando duplicati che prima non venviano individuati;

### net_runner.py
FIle con il compito di inizializzare la rete, allenarla e controllarla.

Nel dettaglio:
- *__get_net*: la rete viene presa dinamicamente in base a ciò che è inserito nel file di configurazione.