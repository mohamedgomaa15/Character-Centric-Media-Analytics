import spacy
from nltk.tokenize import sent_tokenize
import pandas as pd
import sys
import os
import pathlib
from ast import literal_eval

folder_path = pathlib.Path().parent.resolve()
sys.path.append(os.path.join(folder_path,'../'))
from utils import read_data

class NamedEntityRecognizer():

    def __init__(self):
         pass 

    def load_model(self):
        return spacy.load("en_core_web_trf")
    
    def get_ner_inference(self, script):
        sentences = sent_tokenize(script)
        model = self.load_model()
        
        ner_list = []
        for sentence in sentences:
            docs = model(sentence)
            ner_set = set()
            for entity in docs.ent:
                if entity.label_ == "PERSON":
                    first_name = entity.text.split(" ")[0].strip()
                    ner_set.add(first_name)
            
            ner_list.append(ner_set)
        return ner_list
    

    def get_ners(self, dataset_path, save_path=None):

        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df['ners'] = df["ners"].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
            return df
        
        df = read_data(dataset_path)
        df["ners"] = df["script"].apply(self.get_ner_inference)

        if save_path is not None:
            df.to_csv(save_path, index=False)

        return df

    
    #  python -m spacy download en_core_web_trf
