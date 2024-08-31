from net_runner import NetRunner
from config_helper import check_and_get_configuration
from custom_dataset_fruits import CustomDataset
from analyzer import Analyzer


if __name__ == "__main__":
    
    # Carica il file di configurazione, lo valido e ne creo un oggetto a partire dal json.
    cfg_obj = check_and_get_configuration('./config/config.json', './config/config_schema.json')
    
    # Uso un analizzatore per controllare che tutte le classi siano bilanciate
    infoTrainDir = Analyzer(cfg_obj.io.training_folder, debug=False)
    infoValDir = Analyzer(cfg_obj.io.validation_folder, debug=False)
    infoTestDir = Analyzer(cfg_obj.io.test_folder, debug=False)
    
    #TODO: Uso uno script per bilanciare le classi facendo data augmentation [se necessario]

    # Uso un data loader semplicemente per ricavare le classi del dataset.
    classes = CustomDataset(root=cfg_obj.io.training_folder, skip=True, transform=None, debug=True).classes

    # Creo l'oggetto che mi permettera' di addestrare e testare il modello.
    runner = NetRunner(cfg_obj, classes)

    # Se richiesto, eseguo il training.
    if cfg_obj.parameters.train:
        runner.train()

    # Se richiesto, eseguo il test.
    if cfg_obj.parameters.test:
        runner.test(print_acc=True)