from glob import glob
import pandas as pd

 
def read_data(path):
    paths = glob(path+'\*.ass')
    scripts = []
    eposide_names = []
    for path in paths:
        with open(path, 'r') as file:
            lines = file.readlines()
            lines = lines[27:]
            script = ' '.join([','.join(line.split(',')[9:]).strip() for line in lines])

        script = script.replace('\\N', " ")
        eposide_name = int(path.split('-')[-1].split('.')[0].strip())
        scripts.append(script)
        eposide_names.append(eposide_name)

    return pd.DataFrame.from_dict({"eposide": eposide_names, "script": scripts})