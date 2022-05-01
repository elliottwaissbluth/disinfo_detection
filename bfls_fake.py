'''
Scrapes links from a list (sites) and returns a list of top level intra-links (self referring) as a pickle
file in datadir/sources
'''
from pathlib import Path

from custom_command import LinkCountingCommand
from openwpm.command_sequence import CommandSequence, DumpPageSourceCommand
from openwpm.commands.browser_commands import BreadthFirstLinkSearch, GetCommand, BrowseCommand, GetTopLevelLinks
from openwpm.config import BrowserParams, ManagerParams
from openwpm.storage.sql_provider import SQLiteStorageProvider
from openwpm.task_manager import TaskManager
from urls.url_lists import *
import os

def main(first_idx =0):
    try:
        NUM_BROWSERS = 8
        sites = [seed_fake_list]

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
            # Add bot mitigation
            browser_param.bot_mitigation = True

        # Update TaskManager configuration (use this for crawl-wide settings)
        manager_params.data_directory = Path("./datadir/SEED/fake")
        manager_params.log_path = Path("./datadir/openwpm.log")


        # memory_watchdog and process_watchdog are useful for large scale cloud crawls.
        # Please refer to docs/Configuration.md#platform-configuration-options for more information
        # manager_params.memory_watchdog = True
        # manager_params.process_watchdog = True


        # Commands time out by default after 60 seconds
        for site_list in sites:
            with TaskManager(
                manager_params,
                browser_params,
                SQLiteStorageProvider(Path("./datadir/custom-crawl-data.sqlite")),
                None,
            ) as manager:
                # Visits the sites
                for index, site in enumerate(site_list):
                    if index < first_idx:
                        continue
                    def callback(success: bool, val: str = site) -> None:
                        print(
                            f"CommandSequence for {val} ran {'successfully' if success else 'unsuccessfully'}"
                        )

                    # Parallelize sites over all number of browsers set above.
                    command_sequence = CommandSequence(
                        site,
                        site_rank=index,
                        callback=callback,
                    )

                    print('\nSCRAPING SITE {}/{}\n'.format(index, len(large_fake_list)))

                    # Start by visiting the page
                    # command_sequence.append_command(GetCommand(url=site, sleep=3), timeout=60)
                    # Have a look at custom_command.py to see how to implement your own command
                    # command_sequence.append_command(LinkCountingCommand())
                
                    # For manual rescrape
                    command_sequence.append_command(BreadthFirstLinkSearch(site, sleep=3, max_no_increase=10, max_links=100, suffix=index))
                    # CUSTOM
                    #  command_sequence.append_command(BreadthFirstLinkSearch(site, sleep=3, max_no_increase=30, max_links=100, suffix=index))
                    # Run commands across all browsers (simple parallelization)
                    manager.execute_command_sequence(command_sequence)
        return index

    except:
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('ERROR AT SITE IDX = {}'.format(index))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        with open(str(manager_params.data_directory / 'ERRORS.txt'), 'a') as f:
            f.write('DOMAIN_IDX = {}\n'.format(index))
        f.close()
        return index

if __name__ == '__main__':
    # Find first index from sites that have already been scraped
    num_scraped = os.listdir(str(Path.cwd() / 'datadir' / 'SEED' / 'fake' / 'sources'))
    num_scraped = [int(x.split('.')[0]) for x in num_scraped]
    num_scraped = max(num_scraped)

    first_idx = num_scraped-1
    sites_to_scrape = len(seed_fake_list)
    
    print('Scraping from site index {}/{}'.format(first_idx, sites_to_scrape))
    while first_idx < sites_to_scrape:
        first_idx = main(first_idx)
        first_idx += 1
