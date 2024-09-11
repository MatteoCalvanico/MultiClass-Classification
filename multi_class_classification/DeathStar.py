import os
from config_helper import check_and_get_configuration

# Cancella tutte le immagini inserite dal balancer.py
def DeathStar(root_dir, file_pattern="balImg_"):

  for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith(file_pattern):
                filepath = os.path.join(dirpath, filename)
                try:
                    os.remove(filepath)
                    print(f"Deleted: {filepath}")
                except OSError as e:
                    print(f"Error deleting {filepath}: {e}")
      
      
cfg_obj = check_and_get_configuration('./config/config.json', './config/config_schema.json')
  
DeathStar(cfg_obj.io.training_folder)
DeathStar(cfg_obj.io.validation_folder)
DeathStar(cfg_obj.io.test_folder)
