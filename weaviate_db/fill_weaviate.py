import weaviate
from utils.tweets import Tweets
import rfc3339
from tqdm import tqdm

tweets = Tweets(db_file='data/identifier.sqlite',
                remove_hashtags=True,
                remove_urls=True,
                remove_mentions=True,
                remove_nonals=True,
                limit=100000)

# class Tweet:
#     id: int
#     uid: int
#     text: str
#     clean_text: str
#     date: datetime
#     hashtags: list[str]
#     mentions: list[str]
#     urls: list[str]

client = weaviate.Client("http://localhost:8080")

client.schema.delete_all()

schema = {
    "classes": [
        {
            "class": "Tweet",
            "description": "A tweet",
            "properties": [
                {
                    "dataType": ["int"],
                    "description": "id of tweet from nacsos db",
                    "name": "tweetId"
                }, {
                    "dataType": ["text"],
                    "description": "unfiltered text of the tweet",
                    "name": "rawText"
                }, {
                    "dataType": ["text"],
                    "description": "clean text of the tweet",
                    "name": "cleanText"
                }, {
                    "dataType": ["date"],
                    "description": "date of tweet",
                    "name": "publishedDate"
                }, {
                    "dataType": ["Mention"],
                    "description": "users mentioned in this tweet",
                    "name": "hasMentions"
                }, {
                    "dataType": ["Hashtag"],
                    "description": "hashtags used in this tweet",
                    "name": "hasHashtags"
                }

            ]
        }, {
            "class": "User",
            "description": "a twitter user",
            "properties": [
                {
                    "dataType": ["int"],
                    "description": "twitter user id",
                    "name": "userId",
                }, {
                    "dataType": ["Tweet"],
                    "description": "tweets this user wrote",
                    "name": "wroteTweets"
                }
            ]
        }, {
            "class": "Mention",
            "description": "a twitter user",
            "properties": [
                {
                    "dataType": ["string"],
                    "description": "twitter handle",
                    "name": "twitterHandle",
                }, {
                    "dataType": ["Tweet"],
                    "description": "tweets this user was mentioned in",
                    "name": "appearsIn"
                }
            ]
        }, {
            "class": "Hashtag",
            "description": "a twitter hashtag",
            "properties": [
                {
                    "dataType": ["string"],
                    "description": "twitter hashtag",
                    "name": "twitterHashtag",
                }, {
                    "dataType": ["Tweet"],
                    "description": "tweets this hashtag was mentioned in",
                    "name": "appearsIn"
                }
            ]
        }
    ]
}

client.schema.create(schema)

tweet_uuids = []
hashtag_uuids = {}
mention_uuids = {}
user_uuids = {}

for tweet in tqdm(tweets.tweets):
    tweet_uuid = client.data_object.create(
        data_object={
            'tweetId': tweet.id,
            'rawText': tweet.text,
            'cleanText': tweet.clean_text,
            'publishedDate': rfc3339.rfc3339(tweet.date)
        },
        class_name='Tweet')
    tweet_uuids.append(tweet_uuid)

    if tweet.uid in user_uuids:
        user_uuid = user_uuids[tweet.uid]
    else:
        user_uuid = client.data_object.create(
            data_object={
                'userId': tweet.uid
            },
            class_name='User')
    client.data_object.reference.add(from_uuid=user_uuid,
                                     from_property_name='wroteTweets',
                                     to_uuid=tweet_uuid)

    for hashtag in tweet.hashtags:
        if hashtag in hashtag_uuids:
            hashtag_uuid = hashtag_uuids[hashtag]
        else:
            hashtag_uuid = client.data_object.create(
                data_object={
                    'twitterHashtag': hashtag
                },
                class_name='Hashtag')
            hashtag_uuids[hashtag] = hashtag_uuid
        client.data_object.reference.add(from_uuid=tweet_uuid,
                                         from_property_name='hasHashtags',
                                         to_uuid=hashtag_uuid)
        client.data_object.reference.add(from_uuid=hashtag_uuid,
                                         from_property_name='appearsIn',
                                         to_uuid=tweet_uuid)

    for mention in tweet.mentions:
        if mention in mention_uuids:
            mention_uuid = mention_uuids[mention]
        else:
            mention_uuid = client.data_object.create(
                data_object={
                    'twitterHandle': mention
                },
                class_name='Mention')
            mention_uuids[mention] = mention_uuid
        client.data_object.reference.add(from_uuid=tweet_uuid,
                                         from_property_name='hasMentions',
                                         to_uuid=mention_uuid)
        client.data_object.reference.add(from_uuid=mention_uuid,
                                         from_property_name='appearsIn',
                                         to_uuid=tweet_uuid)

# rfc3339.rfc3339(d)
