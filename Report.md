# Model Training Experiments

## Experiment 1: Number of Epochs
In questo esperiemento proveremo a cambiare il numero di *epoch* per controllare come cambiano le performance del modello. L'esperimento è fatto su una *Base CNN*.

| Epochs | Training Accuracy | Validation Accuracy | Training Loss | Notes |
|--------|------------------ |---------------------|---------------|-------|
| 15     |      92.2%        |        93.1%        |   0.0249      | Il target di accuracy impostato è raggiunto all'epoch 5, aumentare il numero di epoch non porterebbe ulteriori benefici |
| 30     |       //          |         //          |     //        |       |
| 50     |       //          |         //          |     //        |       |

Visto che aumentare il numero di epoch si è visto che non servirebbe si può pensare a modificare la rete e il learning rate.

## Experiment 2: Network Architecture
In questo esperiemento useremo diverse *reti* (con num di layer differenti) con pari parametri:
- Epochs: 15,
- Learning Rate: 0.001

| Architecture | Training Accuracy | Validation Accuracy | Notes |
|--------------|-------------------|---------------------|-------|
| Base CNN     |       92.2%       |        93.1%        | Rete di partenza, target accuracy raggiunto alla 5° epoch |
| Deep CNN     |       99.0%       |        99.0%        | Più layers di convoluzione, stesso numero di filtri della Base ma con un layer denso extra nel classificatore, target accuracy raggiunto alla 7° epoch (alla 5° accuracy pari al 95%) |
| Wide CNN     |       98.9%       |        98.7%        | Stesso numero di layer della Base, ma con il doppio dei filtri, target accuracy raggiunto alla 5° epoch, con la prima epoch che aveva una accuracy già al 80%. |

## Experiment 3: Learning Rate
In questo esperiemento modificheremo il *learning rate*. L'esperimento è fatto su una *Base CNN*.

| Learning Rate   | Training Accuracy | Validation Accuracy | Convergence Speed | Notes |
|-----------------|-------------------|---------------------|-------------------|-------|
| 0.001           |      92.2%        |       93.1%         |     5 epochs      | Forse troppo "aggressivo" |
| 0.01            |      5.56%        |       5.56%         |        //         | L'accuracy è diminuita di colpo passando dal 34% (epoch 1) al 5% (epoch 2), probabilmente dovuto al valore troppo alto assegnato al learning rate |
| 0.0001          |      92.9%        |       92.9%         |     13 epochs     | Dopo più del doppio delle epochs la *validation loss* non è migliorata |

Essendo il learning rate alto già di partenza (0.001) aumentarlo di più non ha portato risultati ottimali, diminuirlo invece porta ad una buona accuracy ma con una convergenza molto lenta che quindi non porta veri e propri vantaggi rispetto a quella *0.001*.