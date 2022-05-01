import pandas as pd
import pickle
from pathlib import Path
import os
from bs4 import BeautifulSoup
from bs4.element import Comment
from tqdm import tqdm
from nltk import tokenize
from langdetect import detect
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords

def csv_to_url_list(csv_path, column, outfile, APPEND_HTTPS=True):
    '''
    Reads the csv at csv_path, converts "column" to list, saves to outfile

    if APPEND_HTTPS, prepends 'http://www.' to urls in list
    '''
    # Read csv
    csv = pd.read_csv(csv_path)
    csv_list = csv[column].tolist()

    # Add https
    if APPEND_HTTPS:
        to_pop = []
        for i in range(len(csv_list)):
            if str(csv_list[i]) == 'nan':
                to_pop.append(i)
                continue
            csv_list[i] = 'http://' + str(csv_list[i])
        to_pop.reverse()
        for p in to_pop:
            csv_list.pop(p)
    
    # Save
    with open(outfile, 'wb') as f:
        pickle.dump(csv_list, f)
    f.close()

    # Print
    print('Saved {}'.format(outfile))
    print('Length of list: {}'.format(len(csv_list)))


def generate_link_dataframe(path_to_src_lists, src_list, outfile = None, truncate_lists = False, failed_sites = None, unsaved_sites = None):
    '''
    This function is to be used after the initial breadth first search. It will generate a dictionary which
    can be read by the next openwpm script to visit the sites and construct a directory containing the 
    HTML of all the site in the list.

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ARGUMENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    path_to_src_lists : path or string object
        - This should point to the folder containing the lists to generate the link dictionaries from
        - For example, pointing this to the 'fake' directory will produce the dataframe for the fake dataset
    
    src_list : list
        - List of the top level domains used to create the src_lists in path_to_src_lists

    outfile : str or None
        - Name of the file to save the dataframe under
        - if None, does not save

    truncate_lists : False or int
        - if False, full list of domains in dataframe remains
        - if True, each domain list is truncated to len(<domain list>) <= truncate_lists

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RETURNS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    df : dataframe
    '''

    df = pd.DataFrame(columns=['index', 'site', 'num_links', 'link_dict'])

    if isinstance(path_to_src_lists, str):
        path_to_src_lists = Path(path_to_src_lists)

    for idx, l_list in enumerate([path_to_src_lists / x for x in os.listdir(path_to_src_lists)]):
        
        i = int(str(l_list).split('\\')[-1].split('.')[0])
        if failed_sites is not None:
            if i in failed_sites.keys():
                continue
        if unsaved_sites is not None:
            if i in unsaved_sites.keys():
                continue

        # Open link list
        with open(str(l_list), 'rb') as f:
            link_list = pickle.load(f)
        f.close()

        # Get index of link list

        # Assure that all the lists have links beyond the original
        assert len(link_list) > 1, 'ERROR: No links were gathered from {}'.format(src_list[i])
       

        if truncate_lists:
            link_list = link_list[-1*truncate_lists+1:]

        link_dict = {}
        for idx, link in enumerate(link_list):
            link_dict[idx] = link

        df = df.append(
            {'index' : i,
            'site' : src_list[i],
            'num_links' : len(link_dict),
            'link_dict': link_dict},
            ignore_index=True, 
        )
    
    df = df.set_index('index').sort_index(axis=0)

    # Save and return
    if outfile is not None:
        with open(outfile, 'wb') as f:
            pickle.dump(df, f)
        f.close()

    return df

def check_for_unscraped_sites(path_to_src_lists, src_list):
    '''
    Goes through the scraped source lists and returns the failed websites, along with their indices
    This function could be used to determine which sites to remove from the curated dataset/rescrape
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ARGUMENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    path_to_src_lists : path or string object
        - This should point to the folder containing the lists to generate the link dictionaries from
        - For example, pointing this to the 'fake' directory will produce the dataframe for the fake dataset
    
    src_list : list
        - List of the top level domains used to create the src_lists in path_to_src_lists
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RETURNS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    failed : dict
        - keys = indices of failed sites
        - values = urls of failed sites
    
    unsaved : dict
        - contains list of non-existent files, i.e. openwpm didn't save <index>.pkl
        - keys = indices of unsaved sites
        - values = urls of unsaved sites
    '''
    if isinstance(path_to_src_lists, str):
        path_to_src_lists = Path(path_to_src_lists)
    
    failed = {}
    unsaved = {}

    for i in range(len(src_list)):
        # Check if the file exists
        if not os.path.exists(str(path_to_src_lists / (str(i) + '.pkl'))):
            unsaved[i] = src_list[i]
            continue

        # Open list and see if it contains links beyond home domain
        with open(str(path_to_src_lists / (str(i) + '.pkl')), 'rb') as f:
            link_list = pickle.load(f)
        f.close()

        # Assure that all the lists have links beyond the original
        if not len(link_list) > 1:
            failed[i] = src_list[i]

        

    return failed, unsaved

def evaluate_failed_sites(failed_fake, unsaved_fake, failed_real=None, unsaved_real=None):
    # Print
    print('FAKE FAILED')
    print(list(failed_fake.keys()))
    print(list(failed_fake.values()))
    print('FAKE UNSAVED')
    print(list(unsaved_fake.keys()))
    print(list(unsaved_fake.values()))

    if failed_real is not None and unsaved_real is not None:
        print('REAL FAILED')
        print(list(failed_real.keys()))
        print(list(failed_real.values()))
        print('REAL UNSAVED')
        print(list(unsaved_real.keys()))
        print(list(unsaved_real.values()))

    return

def tag_visible(element):
    '''
    Helper function for populate_all_text()
    
    Takes bs4 element as input and returns boolean indicating whetehr the element is or isn't
    visible text
    '''
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def create_dir(target_directory):
    '''Creates directory target_directory if the directory doesn't already exist'''
    if not os.path.isdir(str(target_directory)):
        os.mkdir(target_directory)

def populate_all_visible_text(source_dir, target_dir):
    '''
    Populates target dir with .txt files of scraped tags from source dir
    Each .txt file contains all visible text

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ARGUMENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    - source_dir
        - source of HTML files, should be raw_sources/<real or fake>/html
    - target_dir
        - directory to save parsed files
    '''
    for dir_ in tqdm([source_dir + '\\' + s for s in os.listdir(source_dir)]):
        files = [dir_ + '\\' + s for s in os.listdir(dir_)]
        dest_dir = target_dir + '\\' + dir_.split('\\')[-1]
        create_dir(dest_dir)
        for html in files:
            # Open HTML
            with open(html, encoding='utf8') as f:
                soup = BeautifulSoup(f, 'html.parser')
            f.close()

            texts = soup.findAll(text=True)
            visible_texts = filter(tag_visible, texts)  
            text = u" ".join(t.strip() for t in visible_texts)

            file_name = html.split('\\')[-1].split('.')[0] + '.txt'
            file_path = dest_dir + '\\' + file_name

            with open(file_path, 'a', encoding='utf8') as f:
                f.writelines([text])
            f.close()
    return

def filter_text(source_dir):
    '''
    Takes the html present in all the .txt files at source dir and returns a list of tokens
    present in those txt files, filterting for non-informative content
    
    ~~~~ ARGUMENTS ~~~~
    - source_dir : path or str
        - directory containing .txt files of parsed text

    ~~~~ RETURNS ~~~~
    - concat : str
        - large string of concatenated text
    '''
    content = ''
    for text in [source_dir / x for x in os.listdir(source_dir)]:
        with open(text, 'r', encoding='utf8') as f:
            lines = f.readlines()
        f.close()

        with open(text, 'r', encoding='utf8') as f:
            page = f.read().replace('\n', ' ')
        f.close()
        
        # Check if lorem ipsum is there, don't append if it is
        if "lorem ipsum" in page:
            continue

        # Check for curly bracket (indicating page is code)
        if '{' in page or '}' in page:
            continue
        
        # Check for at least 5 sentences
        num_sentences = len(tokenize.sent_tokenize(page))
        if num_sentences < 5:
            continue
        # Check for lines that have at least 3 words, not including Javascript
        to_pop = []
        if len(lines) < 2: # Consider "lines" as sentences if lines are only read in as a single line
            lines = tokenize.sent_tokenize(page)
        for i in range(len(lines)):
            if len(lines[i].split(' ')) < 3:
                to_pop.append(i)
            elif lines[i].lower().find('javascript') != -1:
                to_pop.append(i)
        to_pop.reverse()
        for p in to_pop:
            lines.pop(p)
        page_content = ' '.join(lines)

        # Check that the language is English
        try:
            if not detect(page_content) == 'en':
                continue
        except:
            print('no text in {}'.format(text))
            print('page content: {}'.format(page_content))
            continue

        content = content + ' ' + page_content
            

    # Tokenize
    end_punc = ["'", '"', '!', '?', '.', ')']
    tokenized = tokenize.sent_tokenize(content)
    tokenized = list(set(tokenized))
    to_del = []
    for i in range(len(tokenized)):
        if tokenized[i][-1] not in end_punc:
            to_del.append(i)
    to_del.reverse()
    for d in to_del:
        tokenized.pop(d)
    concat = ' '.join(tokenized)

    return concat

def clean_text(text):
    '''
    cleans text of punctuation, strange characters, and numbers
    '''
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    punc_list = '!"#$%&()*+, -./:;<=>?@[\]^_{|}~' + '0123456789'
    t = str.maketrans(dict.fromkeys(punc_list, " "))
    text = text.translate(t)

    # Replace single quote with empty character
    t = str.maketrans(dict.fromkeys("'`", ""))
    text = text.translate(t)

    return text.lower()

def tokenize_dataframe(df):
    '''
    Takes df[col] and tokenizes the string into a list of tokens
    '''
    tokenized = []
    lemmatizer = WordNetLemmatizer()
    porter = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    for _, row in tqdm(df.iterrows()):
        text = row.text
        text = clean_text(text)
        tokens = word_tokenize(text)
        lemstem = []
        for word in tokens:
            lemstem.append(porter.stem(lemmatizer.lemmatize(word)))
        final = [w for w in lemstem if not w in stop_words]

        tokenized.append(final)

    df['tokenized'] = tokenized

    return df
