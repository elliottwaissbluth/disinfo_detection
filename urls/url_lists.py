import pickle
import os

here = os.path.dirname(os.path.abspath(__file__))
real_file = os.path.join(here, 'real.pkl')
fake_file = os.path.join(here, 'fake.pkl')
real_file_100 = os.path.join(here, '100_real_list.pkl')
fake_file_100 = os.path.join(here, '100_fake_list.pkl')
real_file_100_news = os.path.join(here, '100_real_news_list.pkl')
real_file_100_news_df =  os.path.join(here, '100_real_news_df.pkl')
fake_file_100_df =  os.path.join(here, '100_fake_df.pkl')
real_file_500 = os.path.join(here, '500_real_list.pkl')
fake_file_500 = os.path.join(here, '500_fake_list.pkl')
fake_file_500_df = os.path.join(here, '500_fake_df.pkl')
real_file_500_df = os.path.join(here, '500_real_df.pkl')
real_file_large = os.path.join(here, 'large_real_list.pkl')
fake_file_large = os.path.join(here, 'large_fake_list.pkl')
real_file_large_df = os.path.join(here, 'real_df.pkl')
fake_file_large_df = os.path.join(here, 'fake_df.pkl')
seed_fake = os.path.join(here, 'seed_fake_list.pkl')
seed_real = os.path.join(here, 'seed_real_list.pkl')
sp_url_list = os.path.join(here, 'url_list_sp.pkl')


GDI_list_file = os.path.join(here, 'GDI_list.pkl')
GDI_df_file = os.path.join(here, 'GDI_df.pkl')

with open(sp_url_list, 'rb') as f:
    sp_url_list = pickle.load(f)

with open(seed_fake, 'rb') as f:
    seed_fake_list = pickle.load(f)
with open(seed_real, 'rb') as f:
    seed_real_list = pickle.load(f)
    
with open(real_file_large_df, 'rb') as f:
    large_real_df = pickle.load(f)

with open(fake_file_large_df, 'rb') as f:
    large_fake_df = pickle.load(f)

with open(real_file_large, 'rb') as f:
    large_real_list = pickle.load(f)
f.close()

with open(fake_file_large, 'rb') as f:
    large_fake_list = pickle.load(f)
f.close()

with open(real_file, 'rb') as f:
    real_list = pickle.load(f)
f.close()

with open(fake_file, 'rb') as f:
    fake_list = pickle.load(f)
f.close()

with open(real_file_100, 'rb') as f:
    real_list_100 = pickle.load(f)
f.close()

with open(fake_file_100, 'rb') as f:
    fake_list_100 = pickle.load(f)
f.close()

with open(real_file_100_news, 'rb') as f:
    real_list_100_news = pickle.load(f)
f.close()

with open(fake_file_100_df, 'rb') as f:
    fake_list_100_df = pickle.load(f)
f.close()

fake_list_100_df.reset_index(level=0, inplace=True)

with open(real_file_100_news_df, 'rb') as f:
    real_list_100_news_df = pickle.load(f)
f.close()

real_list_100_news_df.reset_index(level=0, inplace=True)

with open(real_file_500, 'rb') as f:
    real_list_500 = pickle.load(f)
f.close()

with open(fake_file_500, 'rb') as f:
    fake_list_500 = pickle.load(f)
f.close()

with open(fake_file_500_df, 'rb') as f:
    fake_500_df = pickle.load(f)
f.close()

fake_500_df.reset_index(level=0, inplace=True)

with open(real_file_500_df, 'rb') as f:
    real_500_df = pickle.load(f)
f.close()

with open(GDI_list_file, 'rb') as f:
    GDI_list = pickle.load(f)
f.close()

with open(GDI_df_file, 'rb') as f:
    GDI_df = pickle.load(f)
f.close()

real_500_df.reset_index(level=0, inplace=True)

test_list = ['https://www.bbc.com/', 'https://www.cdc.gov/']
