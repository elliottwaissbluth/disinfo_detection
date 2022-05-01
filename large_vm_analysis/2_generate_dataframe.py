
from preprocessing_functions import filter_text, tokenize_dataframe
from pathlib import Path
import pandas as pd
import pickle
from tqdm import tqdm

# Loop through the parsed text, fill in dataframe along the way
# NOTE: Change everything
real_dir = Path.cwd().parent / 'html_parsed' / 'real_500'
fake_dir = Path.cwd().parent / 'html_parsed' / 'fake_500'

with open(Path.cwd() / '500_real_list.pkl', 'rb') as f:
    real_list = pickle.load(f)
f.close()

with open(Path.cwd() / '500_fake_list.pkl', 'rb') as f:
    fake_list = pickle.load(f)
f.close()

# This dictionary will be converted into a dataframe
dict_ = {
    'index' : [],
    'site' : [],
    'label' : [],
    'y' : [],
    'text' : [],
}

for idx, site in tqdm(enumerate(real_list)):
    text_dir = real_dir / str(idx)
    text = filter_text(text_dir)

    dict_['index'].append(idx)
    dict_['site'].append(site)
    dict_['label'].append('real')
    dict_['y'].append(0)
    dict_['text'].append(text)

for idx, site in tqdm(enumerate(fake_list)):
    text_dir = fake_dir / str(idx)
    try:
        text = filter_text(text_dir)
    except:
        print('file not found at {}'.format(idx))
        continue

    dict_['index'].append(idx)
    dict_['site'].append(site)
    dict_['label'].append('fake')
    dict_['y'].append(1)
    dict_['text'].append(text)

df = pd.DataFrame.from_dict(dict_)

# Tokenize text
df_tok = tokenize_dataframe(df)

df_tok.head()

# Save dataframe
with open('html_df_tokenized.pkl', 'wb') as f:
    pickle.dump(df_tok, f)
f.close()