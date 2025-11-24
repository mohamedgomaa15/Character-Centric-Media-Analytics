import torch
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
import sys
import os
import pathlib
import nltk

nltk.download('punkt')
#nltk.download('punkt_tap')
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))
from utils import read_data

class ThemeClassifier():

    def __init__(self, theme_list):
        self.model_name = "facebook/bart-large-mnli"
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.theme_list = theme_list
      

    def load_model(self):
        classifier = pipeline("zero-shot-classification",
                        model=self.model_name,
                        device=self.device)
        return classifier

    def theme_classifier_inference(self, script):
      self.them_classifier = self.load_model()
      script_batch = self._script_to_batches(script)
      return self._classify_batches(script_batch)

    def _script_to_batches(script):
        script_sentense = sent_tokenize(script)
        bathch_size = 20
        script_batch = []
        for index in range(0, len(script_sentense), bathch_size):
            batch = script_sentense[index:index+bathch_size]
            script_batch.append(batch)
        return script_batch
        
    def _classify_batches(self, script_batches):
        outputs = self.them_classifier(script_batches, self.theme_list, multi_label=True)
        theme = {}
        for cell in outputs:
            for label, score in zip(cell['labels'], cell['scores']):
                if label not in theme:
                    theme[label] = []
                theme[label].append(score)
        output = {}
        output = {key: np.mean(np.array(value)) for key, value in theme.items()}
        return output
    
    def get_themes(self, dataset_path, save_path=None):
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            return df
        
        df = read_data(dataset_path)

        output_themes = df["scrip"].apply(self.theme_classifier_inference)
        themes_df = pd.DataFrame(output_themes.tolist())
        df[themes_df.columns] = themes_df

        if save_path is not None:
            df.to_csv(save_path, index=False)
    

    
