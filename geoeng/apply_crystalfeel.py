import requests
import json
from utils.tweets import clean_tweet
from tqdm import tqdm

HEADERS = {
    'Host': 'socialanalyticsplus.net',
    'Connection': 'keep-alive',
    'Content-Length': '487',
    'sec-ch-ua': '"Google Chrome";v="95", "Chromium";v="95", ";Not A Brand";v="99"',
    'Accept': '*/*',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'X-Requested-With': 'XMLHttpRequest',
    'sec-ch-ua-mobile': '?0',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36',
    'sec-ch-ua-platform': '"Linux"',
    'Origin': 'https://socialanalyticsplus.net',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Dest': 'empty',
    'Referer': 'https://socialanalyticsplus.net/crystalfeel/',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8,de;q=0.7'
}

if __name__ == '__main__':

    with open('data/geoengineering_tweets_tweets.jsonl') as f, \
            open('data/geoengineering_tweets_tweets_emotions.jsonl', 'w') as f_out:

        for line in tqdm(f):
            tweet = json.loads(line)
            s = clean_tweet(tweet['text'], remove_hashtags=True, remove_urls=True,
                            remove_mentions=True, remove_nonals=False)
            response = requests.post(url='https://socialanalyticsplus.net/crystalfeel/getEmotionScores.php',
                                     data={'tweet': s}, headers=HEADERS)
            try:
                cf = json.loads(response.text)
            except json.decoder.JSONDecodeError:
                cf = response.text

            tweet['crystalfeel'] = cf

            f_out.write(json.dumps(tweet)+'\n')
