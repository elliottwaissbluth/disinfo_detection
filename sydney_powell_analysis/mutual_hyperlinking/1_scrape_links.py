import pickle
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
import os
import tldextract
from pathlib import Path

def scrape_hyperlink_connections(real_dir, fake_dir, domain_df, domain_dict):
    '''
    Parses through the html in source dir, scraping outward referring links along the way.
    Returns a numpy matrix, connections, with link from site i to site j indicated by
    connections[i,j] = 1.

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ARGUMENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    - source_dir
        - source of HTML files, should be raw_sources/<real or fake>/html
    - domain_dict
        - dictionary of domain, index pairs
    '''
    connections = np.zeros(shape=(len(domain_dict), len(domain_dict)))
    skipped_folders = []

    for source_dir in [real_dir, fake_dir]:
        if source_dir == real_dir:
            current_dir = 'real'
        else:
            current_dir = 'fake'

        for dir_ in tqdm([source_dir / s for s in os.listdir(source_dir)]):  # Folders corresponding to domains
            # Get index of domain to access while populating connections
            if current_dir == 'real':
                row = domain_df.loc[(domain_df['label'] == 'real')].loc[(domain_df['index']) == int(dir_.name)]
            else:
                row = domain_df.loc[(domain_df['label'] == 'fake')].loc[(domain_df['index']) == int(dir_.name)]
            
            try:  # some folders exist in sources that aren't in dataframe
                domain_idx = row['condensed_idx'].values[0]
            except:
                print(f'skipping folder {dir_}')
                skipped_folders.append(dir_)
                continue

            files = [dir_ / s for s in os.listdir(dir_)]  # HTML files
            links = []

            # Get links from files
            for html in files:
                # Open HTML
                try:
                    with open(html, encoding='utf8') as f:
                        soup = BeautifulSoup(f, 'html.parser')
                    f.close()
                except Exception as err:
                    print(f'Exception: {err}')
                    print('Could not parse HTML, continuing...')
                    continue
                
                for link in soup.findAll('a', href=True):
                    links.append(link['href'])
            
            # Distill links down to domains
            links = list(set(links))
            domains = [
                        tldextract.extract(l).subdomain + '.' +
                        tldextract.extract(l).domain + '.' +
                        tldextract.extract(l).suffix for l in links
            ]
            domains = list(set(domains))
            filtered_domains = []
            for d in domains:
                if 'www' in d:
                    filtered_domains.append(d.replace('www', ''))
                else:
                    filtered_domains.append(d)
            
            filtered_domains_no_subdomain = []
            unfiltered_domains = []
            # Populate connections
            for d in filtered_domains:
                unfiltered_domains.append(d)
                # If the domain is in the dataset keys outright
                if d in domain_dict.keys():
                    connections[domain_idx, domain_dict[d]] = 1
                if not d.split('.')[-1]:
                    continue
                # If there is a prefix to the domain and less than 4 domain elements
                # i.e. web.domain.co.uk would be 4, web.domain.com would be 3
                # Sometimes sites have multiple countries prepended, like de.vexels.com and br.vexels.com
                if d.split('.')[0] and len(d.split('.')) < 4:
                    if '.' + d.split('.')[1] + d.split('.')[2] in domain_dict.keys():
                        connections[domain_idx, domain_dict['.' + d.split('.')[1] + d.split('.')[2]]] = 1
                    elif '.' + d.split('.')[0] + '.' + d.split('.')[2] in domain_dict.keys():
                        connections[domain_idx, domain_dict['.' + d.split('.')[0] + '.' + d.split('.')[2]]] = 1
                        filtered_domains_no_subdomain.append('.' + d.split('.')[0] + '.' + d.split('.')[2])
                    else:
                        filtered_domains_no_subdomain.append('.' + d.split('.')[1] + '.' + d.split('.')[2])
                # If there is a prefix and the domain has 4 elements, i.e. web.domain.co.uk
                # We're only interested in selecting domain.co.uk
                elif d.split('.')[0] and len(d.split('.')) == 4:
                    if '.' + d.split('.')[1] + '.' + d.split('.')[2] + '.' + d.split('.')[3] in domain_dict.keys():
                        connections[domain_idx, domain_dict['.' + d.split('.')[1] + '.' + d.split('.')[2] + '.' + d.split('.')[3]]] = 1
                        filtered_domains_no_subdomain.append('.' + d.split('.')[1] + '.' + d.split('.')[2] + '.' + d.split('.')[3])
                    elif '.' + d.split('.')[0] + d.split('.')[2] + d.split('.')[3] in domain_dict.keys():
                        connections[domain_idx, domain_dict['.' + d.split('.')[0] + d.split('.')[2] + d.split('.')[3]]] = 1
                        filtered_domains_no_subdomain.append('.' + d.split('.')[0] + '.' + d.split('.')[2] + '.' + d.split('.')[3])
                    else:
                        filtered_domains_no_subdomain.append('.' + d.split('.')[1] + '.' + d.split('.')[2] + '.' + d.split('.')[3])
                # If there is not a prefix and the domain has 3 elements, i.e. .facebook.com
                # Ignore empty lists
                elif d.split('.') != ['', '', ''] and len(d.split('.')) < 4:
                    if '.' + d.split('.')[1] + '.' + d.split('.')[2] in domain_dict.keys():
                        connections[domain_idx, domain_dict['.' + d.split('.')[1] + '.' + d.split('.')[2]]] = 1
                        filtered_domains_no_subdomain.append('.' + d.split('.')[1] + '.' + d.split('.')[2])
                    else:
                        filtered_domains_no_subdomain.append(d)
                elif d.split('.') != ['', '', '', ''] and len(d.split('.')) == 4:
                    if '.' + d.split('.')[1] + '.' + d.split('.')[2] + '.' + d.split('.')[3] in domain_dict.keys():
                        connections[domain_idx, domain_dict['.' + d.split('.')[1] + '.' + d.split('.')[2] + '.' + d.split('.')[3]]] = 1
                        filtered_domains_no_subdomain.append('.' + d.split('.')[1] + '.' + d.split('.')[2] + '.' + d.split('.')[3])
                    else:
                        filtered_domains_no_subdomain.append(d)


            filtered_domains_no_subdomain = list(set(filtered_domains_no_subdomain))
            with open(Path.cwd() / 'full_analysis' / 'total_domains' / (str(domain_idx) + '.pkl'), 'wb') as f:
                pickle.dump(filtered_domains_no_subdomain, f)

    return connections, skipped_folders


real_dir = Path.cwd().parent / 'large_vm_analysis' / 'raw_sources' / 'real' / 'sources'
fake_dir = Path.cwd().parent / 'large_vm_analysis' / 'raw_sources' / 'fake' / 'sources'

with open(Path.cwd() / 'full_analysis' / 'domain_df.pkl', 'rb') as f:
    df = pickle.load(f)

domain_dict = {}
for _, row in df.iterrows():
    domain_dict[row.domain.replace('www', '')] = row['condensed_idx']
connections, skipped_folders = scrape_hyperlink_connections(real_dir, fake_dir, df, domain_dict)

with open(Path.cwd() / 'full_analysis' / 'connections_new.pkl', 'wb') as f:
    pickle.dump(connections, f)

with open(Path.cwd() / 'full_analysis' / 'domain_dict_new.pkl', 'wb') as f:
    pickle.dump(domain_dict, f)

with open(Path.cwd() / 'full_analysis' / 'skipped_folders_new.pkl', 'wb') as f:
    pickle.dump(skipped_folders, f)

print(f'skipped {len(skipped_folders)} folders')
