MODELS = {
    # to find more models, browse this page:
    # https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads
    # Hint: the search function doesn't really work...
    "cardiff-sentiment": {
        # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/sentiment/mapping.txt
        "model": "cardiffnlp/twitter-roberta-base-sentiment",
        "labels": ["negative", "neutral", "positive"],
    },
    "cardiff-emotion": {
        # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/emotion/mapping.txt
        "model": "cardiffnlp/twitter-roberta-base-emotion",
        "labels": ["anger", "joy", "optimism", "sadness"],
    },
    "cardiff-offensive": {
        # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/offensive/mapping.txt
        "model": "cardiffnlp/twitter-roberta-base-offensive",
        "labels": ["not-offensive", "offensive"],
    },
    "cardiff-stance-climate": {
        # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/stance/mapping.txt
        "model": "cardiffnlp/twitter-roberta-base-stance-climate",
        "labels": ["none", "against", "favor"],
    },
    "geomotions-orig": {
        # https://huggingface.co/monologg/bert-base-cased-goemotions-original/blob/main/config.json
        "model": "monologg/bert-base-cased-goemotions-original",
        "labels": [
            "admiration",
            "amusement",
            "anger",
            "annoyance",
            "approval",
            "caring",
            "confusion",
            "curiosity",
            "desire",
            "disappointment",
            "disapproval",
            "disgust",
            "embarrassment",
            "excitement",
            "fear",
            "gratitude",
            "grief",
            "joy",
            "love",
            "nervousness",
            "neutral",
            "optimism",
            "pride",
            "realization",
            "relief",
            "remorse",
            "sadness",
            "surprise",
        ],
    },
    "geomotions-ekman": {
        # https://huggingface.co/monologg/bert-base-cased-goemotions-ekman/blob/main/config.json
        "model": "monologg/bert-base-cased-goemotions-ekman",
        "labels": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    },
    # 'nlptown-sentiment': {
    #     # https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment/blob/main/config.json
    #     'model': 'nlptown/bert-base-multilingual-uncased-sentiment',
    #     'labels': ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
    # },
    "bertweet-sentiment": {
        # https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis
        "model": "finiteautomata/bertweet-base-sentiment-analysis",
        "labels": ["negative", "neutral", "positive"],
    },
    "bertweet-emotions": {
        # https://huggingface.co/finiteautomata/bertweet-base-emotion-analysis
        "model": "finiteautomata/bertweet-base-emotion-analysis",
        "labels": ["others", "joy", "sadness", "anger", "surprise", "disgust", "fear"],
    },
    # 'bert-sst2': {
    #     # https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/blob/main/config.json
    #     'model': 'distilbert-base-uncased-finetuned-sst-2-english',
    #     'labels': ['negative', 'positive']
    # }
}