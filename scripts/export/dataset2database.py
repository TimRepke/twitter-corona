import sqlite3
import json
from tqdm import tqdm
from pathlib import Path

DUMP_FILE = 'viewer/dump_7000000_monthly_downsampled2.json'
DATA = 'data/climate2/tweets_filtered_annotated_all.jsonl'

TARGET_DB = '../topic-explorer/data/test1/data.sqlite'

dbf = Path(TARGET_DB)
if dbf.exists():
    dbf.unlink()

con = sqlite3.connect(TARGET_DB)
con.row_factory = sqlite3.Row
cur = con.cursor()

cur.executescript('''
CREATE TABLE topics (
    id          INTEGER PRIMARY KEY NOT NULL,
    name        TEXT,
    terms_tfidf TEXT,
    terms_mmr   TEXT
);

CREATE TABLE meta_topics (
    id          INTEGER PRIMARY KEY NOT NULL,
    name        TEXT            NOT NULL,
    description TEXT
);

CREATE TABLE similarities (
    topic_a    INTEGER  NOT NULL,
    topic_b    INTEGER  NOT NULL,
    metric     TEXT NOT NULL,
    similarity REAL NOT NULL,
    UNIQUE (topic_a, topic_b, metric) ON CONFLICT REPLACE,
    FOREIGN KEY (topic_a) REFERENCES topics (id),
    FOREIGN KEY (topic_b) REFERENCES topics (id)
);
CREATE INDEX ix_similarities_topic_a ON similarities (topic_a);

CREATE TABLE tweets (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    nacsos_id  TEXT NOT NULL,
    twitter_id TEXT NOT NULL,
    author_id  TEXT NOT NULL,
    txt        TEXT NOT NULL,
    created    TEXT NOT NULL,
    topic      INTEGER,
    meta_topic INTEGER,
    retweets   INTEGER  NOT NULL,
    likes      INTEGER  NOT NULL,
    replies    INTEGER  NOT NULL,
    group_year CHAR(4),
    group_month CHAR(7),
    group_day  CHAR(10),
    x          FLOAT NOT NULL,
    y          FLOAT NOT NULL,
    FOREIGN KEY (topic) REFERENCES topics (id),
    FOREIGN KEY (meta_topic) REFERENCES meta_topics (id)
);
CREATE INDEX ix_tweets_nacsos_id ON tweets (twitter_id);
CREATE INDEX ix_tweets_twitter_id ON tweets (twitter_id);
CREATE INDEX ix_tweets_topic ON tweets (topic);
CREATE INDEX ix_tweets_meta_topic ON tweets (meta_topic);
CREATE INDEX ix_tweets_group_year ON tweets (group_year);
CREATE INDEX ix_tweets_group_month ON tweets (group_month);
CREATE INDEX ix_tweets_group_day ON tweets (group_day);

CREATE VIRTUAL TABLE tweets_fts USING fts5(txt, content=tweets, content_rowid=id);
''')
con.commit()

with open(DUMP_FILE, 'r') as f:
    print('loading dump')
    dump = json.load(f)

    print('inserting topics')
    data = [{'id': ti, 'name': None, 'terms_tfidf': topic.get('tfidf'), 'terms_mmr': topic.get('mmr')}
            for ti, topic in enumerate(dump['topics'])]
    cur.executemany('INSERT INTO topics (id, name, terms_tfidf, terms_mmr) '
                    'VALUES(:id, :name, :terms_tfidf, :terms_mmr);', data)
    con.commit()

    print('inserting similarities')
    data = [
        {'topic_a': tia, 'topic_b': topic_b[0], 'metric': space_name, 'similarity': topic_b[1]}
        for space, space_name in [('ld', '2d-l2'), ('hd', 'emb-cos')]
        for tia, neighbours in enumerate(dump['neighbours'])
        for topic_b in neighbours[space]
    ]
    cur.executemany('INSERT INTO similarities (topic_a, topic_b, metric, similarity) '
                    'VALUES(:topic_a, :topic_b, :metric, :similarity);', data)
    con.commit()

    print('inserting meta topics')
    cur.execute("INSERT INTO meta_topics VALUES(1, 'Not Relevant', NULL);")
    cur.execute("INSERT INTO meta_topics VALUES(2, 'COVID-19', NULL);")
    cur.execute("INSERT INTO meta_topics VALUES(3, 'Politics', NULL);")
    cur.execute("INSERT INTO meta_topics VALUES(4, 'Movements', NULL);")
    cur.execute("INSERT INTO meta_topics VALUES(5, 'Impacts', NULL);")
    cur.execute("INSERT INTO meta_topics VALUES(6, 'Causes', NULL);")
    cur.execute("INSERT INTO meta_topics VALUES(7, 'Solutions', NULL);")
    cur.execute("INSERT INTO meta_topics VALUES(8, 'Contrarian', NULL);")
    cur.execute("INSERT INTO meta_topics VALUES(9, 'Other', NULL);")
    con.commit()


def line2tup(li, ln):
    tweet = json.loads(ln)

    topic = tweet['t_km']
    # if topic < 0:
    #     topic = 0

    meta_topic = 9
    if tweet['st_nr'] > 0:
        meta_topic = 1
    elif tweet['st_cov'] > 0:
        meta_topic = 2
    elif tweet['st_pol'] > 0:
        meta_topic = 3
    elif tweet['st_mov'] > 0:
        meta_topic = 4
    elif tweet['st_imp'] > 0:
        meta_topic = 5
    elif tweet['st_cau'] > 0:
        meta_topic = 6
    elif tweet['st_sol'] > 0:
        meta_topic = 7
    elif tweet['st_con'] > 0:
        meta_topic = 8
    # elif tweet['st_oth'] > 0:
    #     meta_topic = 9

    return {
        'nacsos_id': str(li),
        'twitter_id': tweet['id'],
        'author_id': tweet['author_id'],
        'txt': tweet['text'],
        'created': tweet['created_at'],
        'topic': topic,
        'meta_topic': meta_topic,
        'retweets': tweet['retweets_count'],
        'likes': tweet['favorites_count'],
        'replies': tweet['replies_count'],
        'group_year': tweet['created_at'][:4],
        'group_month': tweet['created_at'][:7],
        'group_day': tweet['created_at'][:10],
        'x': tweet['x'],
        'y': tweet['y']
    }


def read_batched():
    batch_ = []
    with open(DATA, 'r') as f_in:
        for li, line in enumerate(f_in):
            batch_.append((li, line))
            if li % 100000 == 0:
                yield batch_
                batch_ = []
        yield batch_


print('inserting data')
for batch in tqdm(read_batched()):
    data = [line2tup(li_, ln_) for li_, ln_ in batch]
    cur.executemany('INSERT INTO tweets (nacsos_id, twitter_id, author_id, txt, created, topic, meta_topic, retweets, likes, replies, group_year, group_month, group_day, x, y) '
                    'VALUES (:nacsos_id, :twitter_id, :author_id, :txt, :created, :topic, :meta_topic, :retweets, :likes, :replies, :group_year, :group_month, :group_day, :x, :y);', data)
    con.commit()

con.close()

print('done!')
