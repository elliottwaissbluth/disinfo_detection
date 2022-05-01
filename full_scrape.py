from pathlib import Path

from numpy.lib.shape_base import column_stack
from custom_command import LinkCountingCommand
from openwpm.command_sequence import CommandSequence, DumpPageSourceCommand
from openwpm.commands.browser_commands import BreadthFirstLinkSearch, GetCommand, BrowseCommand, GetTopLevelLinks
from openwpm.config import BrowserParams, ManagerParams
from openwpm.storage.sql_provider import SQLiteStorageProvider
from openwpm.task_manager import TaskManager
from urls.url_lists import *
import os
import time
import httpx
import asyncio
import cProfile
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

async def get_links_from_urls(sites): 
    '''Takes a list of sites and returns the GET response from each one

    Args:
        sites (list of str): each string should be a unique url to visit

    Returns:
        list : list of responses, HTML can be gathered using list[<index>].text
    '''
    async with httpx.AsyncClient() as client:
        try:
            tasks = (client.get(url) for url in sites)
            reqs = await asyncio.gather(*tasks)
        except asyncio.exceptions.CancelledError:
            if tasks:
                print('Cancelled task, cancelling {} child tasks'.format(len(tasks)))
                for task in tasks:
                    task.cancel()
            else:
                print('Cancelled')
            raise
    return reqs
    
def get_intra_referring_links_from_html(html, domain):
    '''Scrapes the intra-referring links from the given html and appends the new ones to links

    Args:
        html (str): HTML from scraped url, should be scraped from subdomain of domain
        domain (str): the root domain, used to find links containing the root
        
    Returns:
         list: contains all intra-referring links present in html
    '''
    soup = BeautifulSoup(html, features="html.parser")
    a_tags = soup.findAll("a", href=True)
    links = []
    if domain[-1] == '/':
        domain = domain[:-1]
    for a in a_tags:
        link = a.get("href")
        if "http" not in link and '/' in link:
            links.append(domain+link)
        elif domain in link:
            links.append(link)
    return list(set(links))

def initialize_dataframe(sites):
    '''Initializes the DataFrame which holds all the scraped links, the DataFrame will also serve as a cache which
    the scraper will pull from to decide which sites to scrape next

    Args:
        sites (list of strings): the original list of sites to scrape

    Returns:
        DataFrame: Initial DataFrame which may be read for scraping
    '''
    # Create a DataFrame to hold all the scraped links
    # First, generate the DataFrame from the original list
    df = pd.DataFrame(columns=['domain', 'visited', 'failed', 'next', 'num_links', 'links'])
    df['domain'] = sites                                                # domain (str): the original domain
    df['visited'] = [[] for _ in sites]                                 # visited (list): list of visited domains
    df['failed'] = [[] for _ in sites]                                  # failed (list): list of domains that threw an error response
    df['next'] = sites                                                  # next (str): next url to visit
    df['links'] = [[x] for x in sites]                                  # links (list): list of scraped domains
    df['num_links'] = df.apply(lambda row: len(row.links), axis=1)      # num_links (int): number of urls in links
    return df

def reload_dataframe(df, visited_sites, reqs, status, max_links=100):
    df['temp_column'] = None  # Use temporary column to hold new list values
    # Append to visited if successful, else append to failed
    for i, (k, v) in enumerate(visited_sites.items()):
        if not status[k]:  # If status is False, append site to failed and copy links column into temp
            df.loc[k, 'failed'].append(v)
            df.loc[df.index == k, 'temp_column'] = pd.Series([df.loc[k, 'links']], index=df.loc[df.index == k].index, dtype='object')
        else:              # Otherwise, mark it as visited
            df.loc[k, 'visited'].append(v)
            
            # Get intra-referring links
            links = get_intra_referring_links_from_html(html=reqs[i].text, domain=df.loc[k, 'domain'])

            # Append the newly scraped links to the links column, do not add repeats
            df.loc[df.index == k, 'temp_column'] = pd.Series([df.loc[k, 'links'] + [x for x in links if x not in df.loc[k, 'links']]], index=df.loc[df.index == k].index, dtype='object')
            assert len(df.loc[k, 'temp_column']) == len(set(df.loc[k, 'temp_column'])), 'ERROR: There are duplicates added to the scraped links list'
            
            # Update num_links
            df.loc[k, 'num_links'] = len(df.loc[k, 'temp_column'])
            
            # Move on to the next site to visit
            if df.loc[k, 'num_links'] >= max_links:
                df.loc[k, 'next'] = np.nan
            else:
                next_site_index = len(df.loc[k, 'failed']) + len(df.loc[k, 'visited'])
                assert next_site_index <= len(df.loc[k, 'links']), 'ERROR: Ran out of sites to scrape for domain {}'.format(v)
                df.loc[k, 'next'] = df.loc[k, 'temp_column'][next_site_index]
            
    # Turn the temp column into the "links" column
    df = df.drop(columns=['links']).rename(columns={'temp_column' : 'links'})
    return df
            
def generate_next_sites(df):
    df_next = df[df.next.notnull()] # when a site is done scraping, set this to NaN so it will be ignored
    return dict(zip(df_next.index, df_next.next))

def main():
    sites = real_list_100_news
    df = initialize_dataframe(sites)
    print(df.head())

    next_sites = generate_next_sites(df)
    print('NEXT SITES: ')
    print(next_sites)
    
    while len(next_sites) > 0:
        # Scrape the next sites
        reqs = asyncio.run(get_links_from_urls(next_sites.values()))        # Get the html from the original set of urls
        status = dict(zip(next_sites.keys(), [x.is_success for x in reqs])) # {index (int) : STATUS (bool)}, False if failed to perform request

        # Reload the dataframe for the next round of scraping
        df = reload_dataframe(df, visited_sites=next_sites, reqs=reqs, status=status)
        print(status)
        print(df)
         
        next_sites = generate_next_sites(df)
        print('NEXT SITES: ')
        print(next_sites)

if __name__ == '__main__':
    main()
    
    
