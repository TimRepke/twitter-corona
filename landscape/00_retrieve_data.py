import json
import psycopg
from psycopg.rows import dict_row
from tqdm import tqdm
import os

TARGET = 'data/climate_tweets.jsonl'

with psycopg.connect(f'postgresql://{os.environ["DB_USER"]}:{os.environ["DB_PASS"]}@'
                     f'{os.environ["DB_HOST"]}:{os.environ["DB_PORT"]}/{os.environ["DB_NAME"]}',
                     row_factory=dict_row) as conn:
    # Open a cursor to perform database operations
    with conn.cursor(row_factory=dict_row) as cur, open(TARGET, 'w') as f_out:
        print('run query')
        cur.execute("""
        SELECT ts.*, tt.*
        FROM twitter_status_searches tss
        LEFT JOIN twitter_status ts on tss.status_id = ts.twitterbasemodel_ptr_id
        LEFT JOIN twitter_twitterbasemodel tt on ts.twitterbasemodel_ptr_id = tt.id
        WHERE twittersearch_id=83
        ORDER BY tt.created_at;""")

        print('writing records')
        for record in tqdm(cur):
            record['created_at'] = record['created_at'].isoformat()
            record['fetched'] = record['fetched'].isoformat()
            f_out.write(json.dumps(record) + '\n')
