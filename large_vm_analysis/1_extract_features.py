
from preprocessing_functions import *

# NOTE:
# Change these after importing data to vm

# real_source_dir = str(Path.cwd().parent / 'raw_sources' / 'real_html' / 'sources')
# real_target_dir = str(Path.cwd().parent / 'html_parsed' / 'real_500')
# fake_source_dir = str(Path.cwd().parent / 'raw_sources' / 'fake_html' / 'sources')
# fake_target_dir = str(Path.cwd().parent / 'html_parsed' / 'fake_500')

populate_all_visible_text(real_source_dir, real_target_dir)
populate_all_visible_text(fake_source_dir, fake_target_dir)