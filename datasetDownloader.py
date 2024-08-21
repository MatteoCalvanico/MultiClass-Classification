# File needed for the dowload the dataset (moltean/fruits: https://www.kaggle.com/datasets/moltean/fruits) from Kaggle

import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi


def checkFolder(path):
    """ Return True if a folder do not exist or if is empty """
    
    if os.path.exists(path):
        return len( os.listdir(path)) == 0
    else:
        return True




if __name__ == "__main__":
    
    os.environ['KAGGLE_USERNAME'] = 'matteocalvanico'
    os.environ['KAGGLE_KEY'] = '2c941be0e8b7029e2b4aa07d87c40478'

    api = KaggleApi()
    api.authenticate()
    # If Kaggle doesn't find the environment variables go to: C:\Users\<name>\.kaggle 
    # and create a json file named kaggle.json 
    # and put this inside: {"username":"matteocalvanico","key":"2c941be0e8b7029e2b4aa07d87c40478"}

    dataset = 'moltean/fruits'
    path = 'dataset'


    if checkFolder("./dataset"):
        print("Starting download...")
        api.dataset_download_files(dataset, path, unzip=True)
        print("Download ended")
    else:
        print("Dataset already installed, download blocked")
