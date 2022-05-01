from bs4 import BeautifulSoup
import pandas as pd
import os
from pathlib import Path
import numpy as np
import pickle
from tqdm import tqdm

def html_looper(func):

    # Get url keys
    with open('real_url_key.pkl', 'rb') as f:
        real_url_key = pickle.load(f)
    f.close()
    with open('fake_url_key.pkl', 'rb') as f:
        fake_url_key = pickle.load(f)
    f.close()

    # Replace these directories with folders containing HTML files
    directory1 = Path.cwd().parent / 'HTML' / 'real'
    directory2 = Path.cwd().parent / 'HTML' / 'fake'
    directory = [directory1, directory2]

    dict_ = {'index' : [],
            'label' : [],
            'site' : []}

    for dir_ in tqdm(directory):
        for html in os.listdir(dir_):
            # Open html file as soup object
            with open(str(dir_) + '/' + html, encoding='utf8') as f:
                soup = BeautifulSoup(f, 'html.parser')
            f.close()

            # Populate dict
            # the html files are saved as <index>.html
            index = int(str(html).split('.')[0]) 
            # extracts the string 'real' or 'fake' at end of directory
            label = str(dir_)[-4:]   
            # aligns the index to the url using real_url_key and fake_url_key
            site = real_url_key[index] if str(dir_)[-4:] == 'real' else fake_url_key[index]

            dict_['index'].append(index)
            dict_['label'].append(label)
            dict_['site'].append(site)

            func(soup, dict_)

    for k in dict_.keys():
        if len(dict_[k]) == 0:
            print('key "{}" has length 0, deleting'.format(k))
            del dict_[k]
    
    return dict_

def if_not_key(dict_, key):
    '''
    takes a list, key, and puts empty lists in dict_ corresponding to all k in key
    '''
    for k in key:
        if not k in dict_:
            print('dict_ does not have key {}, creating...'.format(k))
            dict_[k] = []
    return

def dict_2_csv(dict_, csv_name):
    '''
    Creates a merged CSV from a dictionary
    '''
    df = pd.DataFrame(dict_)
    df_csv = pd.read_csv('html_data_expanded.csv')
    joined = df_csv.merge(df, on=['index', 'label'])
    joined.rename(columns={'Column1' : 'global_index', 'site_x' : 'site'}, inplace=True)
    joined.drop(columns=['site_y'], inplace=True)
    joined.to_csv(csv_name)
    return

# ---------------------------------------------------------------------------- #
#                          CUSTOM INTRA-LOOP FUNCTIONS                         #
# ---------------------------------------------------------------------------- #

def get_metatags(soup, dict_):

    if not dict_.has_key('meta_content_combined'):
        dict_['meta_content_combined'] = []
        
    metatags = soup.find_all('meta')
    content = ''
    for tag in metatags:
        try:
            content += ' ' + tag.attrs['content']
        except:
            try:
                content += ' ' + tag.attrs['value']
            except:
                continue

    dict_['meta_content_combined'].append(content)