'''
Takes a dataframe with columns ['index', 'site', 'num_links', 'list'], creates a folder titled <index> in
datadir/sources, populates that folder with the entire scraped HTML of <index of list element>.html and 
a screenshot with the same title. Note: dataframe['list'] contains dictionaries of the form {index : site} for
all scraped sites.
'''
from pathlib import Path
import pandas as pd
import pickle
import os
import time

from custom_command import LinkCountingCommand
from openwpm.command_sequence import CommandSequence, DumpPageSourceCommand
from openwpm.commands.browser_commands import CustomScreenshotFullPageCommand, GetCommand, BrowseCommand, GetTopLevelLinks
from openwpm.config import BrowserParams, ManagerParams
from openwpm.storage.sql_provider import SQLiteStorageProvider
from openwpm.task_manager import TaskManager
from urls.url_lists import *

idx = 0

def main(start_idx=0):
    try:
        # Change this to change the dataframe used for this program
        df = large_real_df
        df = df.reset_index()

        def timer(start, end):
            hours, rem = divmod(end-start, 3600)
            minutes, seconds = divmod(rem, 60)
            return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

        # The list of sites that we wish to crawl
        NUM_BROWSERS = 8

        # Loads the default ManagerParams
        # and NUM_BROWSERS copies of the default BrowserParams

        manager_params = ManagerParams(num_browsers=NUM_BROWSERS)
        browser_params = [BrowserParams(display_mode="headless") for _ in range(NUM_BROWSERS)]

        # Update browser configuration (use this for per-browser settings)
        for browser_param in browser_params:
            # Record HTTP Requests and Responses
            browser_param.http_instrument = False
            # Record cookie changes
            browser_param.cookie_instrument = False
            # Record Navigations
            browser_param.navigation_instrument = False
            # Record JS Web API calls
            browser_param.js_instrument = False
            # Record the callstack of all WebRequests made
            browser_param.callstack_instrument = False
            # Record DNS resolution
            browser_param.dns_instrument = False
            # Turn on bot mitigation
            browser_param.bot_mitigation = True

        # Update TaskManager configuration (use this for crawl-wide settings)
        manager_params.data_directory = Path("./datadir/LARGE/html/real")
        manager_params.log_path = Path("./datadir/openwpm.log")

        # memory_watchdog and process_watchdog are useful for large scale cloud crawls.
        # Please refer to docs/Configuration.md#platform-configuration-options for more information
        # manager_params.memory_watchdog = True
        # manager_params.process_watchdog = True

        # total_sites_to_scrape = df['num_links'].sum()
        sites_scraped = 0
        for idx ,row in df.iterrows():
            if idx > start_idx:
                break
            sites_scraped += min(100, len(row['link_dict']))
#             for _,_ in row['link_dict'].items():
                # sites_scraped += 1
        total_sites_to_scrape = 0
        for idx, row in df.iterrows():
            total_sites_to_scrape += min(100, len(row['link_dict']))

        # total_sites_to_scrape = len(df)*100
        
        # Commands time out by default after 60 seconds
        with TaskManager(
            manager_params,
            browser_params,
            SQLiteStorageProvider(Path("./datadir/custom-crawl-data.sqlite")),
            None,
        ) as manager:

            # sites_scraped = len(df.loc[:start_idx,])
            start = time.time()

            # Visits the sites
            for index, row in df.iterrows():
                inner_idx = index
                if row['index'] < start_idx:
                    continue
                
                # Create directory to hold data if one does not already exist
                if not os.path.isdir(os.path.join(manager_params.data_directory / 'sources', str(row['index']))):
                    os.mkdir(os.path.join(manager_params.data_directory / 'sources', str(row['index'])))
                else:
                    print('\nDOMAIN_IDX = {}, DOMAIN = {} has already been scraped, continuing\n'.format(inner_idx, row['index']))
                    sites_scraped += 100
                    continue

                for index, site in row['link_dict'].items():
                    if index > 100:
                        continue
                    def callback(success: bool, val: str = site) -> None:
                        print(
                            f"CommandSequence for {val} ran {'successfully' if success else 'unsuccessfully'}"
                        )
                    end = time.time()
                    print('\nSCRAPING SITE {}/{}, DOMAIN_IDX = {}, DOMAIN = {}\nTime Elapsed: {}\n'.format(sites_scraped, total_sites_to_scrape, inner_idx, row['index'], timer(start,end)))
                    
                    sites_scraped += 1

                    # Parallelize sites over all number of browsers set above.
                    command_sequence = CommandSequence(
                        site,
                        site_rank=index,
                        callback=callback,
                    )

                    # Start by visiting the page
                    command_sequence.append_command(GetCommand(url=site, sleep=3), timeout=60)

                    # Get content as HTML file
                    command_sequence.append_command(DumpPageSourceCommand(str(index), dir=str(row['index'])))

                    # Run commands across all browsers (simple parallelization)
                    manager.execute_command_sequence(command_sequence)
            return inner_idx
    except:
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('ERROR AT SITE IDX = {}'.format(inner_idx))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        with open(str(manager_params.data_directory / 'sources' / 'ERRORS.txt'), 'a') as f:
            f.write('DOMAIN_IDX = {}, DOMAIN = {}'.format(inner_idx, row['index']))
        f.close()
        return inner_idx

if __name__ == '__main__':
    first_idx = 0
    sites_to_scrape = len(large_real_df)

    while first_idx < sites_to_scrape:
        first_idx = main(first_idx)
        first_idx += 1  # increment to go on to next site
        
