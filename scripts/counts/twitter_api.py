import os
import json
import logging
from searchtweets import ResultStream

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', level=logging.DEBUG)
NAME = 'cc_Jan2017_Sept2022'
QUERY = '"climate change" lang:en -is:retweet -is:quote'

stream = ResultStream(
    endpoint='https://api.twitter.com/2/tweets/counts/all',
    request_parameters={
        'query': QUERY,
        'start_time': '2017-01-01T00:00:00Z',
        'end_time': '2022-10-10T23:59:59Z',
        'granularity': 'day'
    },
    bearer_token=os.getenv("TWITTER"),
    output_format='r')

logging.info(f'Searching for: {QUERY}')
FILE = f'data/counts/{NAME}.jsonl'
logging.info(f'Writing to: {FILE}')
if os.path.exists(FILE):
    raise FileExistsError('File already exists. If you are certain you want to proceed, rename or delete the file.')

with open(FILE, 'w') as f_out:
    for results in stream.stream():
        logging.info('Received page.')
        if 'data' in results and type(results['data']) == list:
            f_out.write(json.dumps(results) + '\n')
        else:
            logging.error('Something went wrong!')
