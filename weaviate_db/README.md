1. Get docker-compose config (or use the one here):  
   https://www.semi.technology/developers/weaviate/current/getting-started/installation.html
2. Spin up the container:  
   `docker-compose up`
3. Populate the database:   
   run `fill_weaviate.py` from parent directory
4. Open the console and play around:   
   http://console.semi.technology/console/query

```
{
  Get{
    Mention(
      where:{
        operator: Equal,
        path:["twitterHandle"],
        valueString:"@BillNye"
      }
    ){
      twitterHandle
      appearsIn {
        ... on Tweet {
          rawText
        }
      }
    }
  }
}
```
```
{
  Get{
    Hashtag(
      where:{
        operator: Equal,
        path:["twitterHashtag"],
        valueString:"#brexit"
      }
    ){
      
      appearsIn {
        ... on Tweet {
          rawText
        }
      }
    }
  }
}
```
```
{
  Get {
    Tweet(
      ask: {
        question: "Who is the president?",
        properties: ["cleanText"]
      }, 
      limit: 5
    ) {
      cleanText
      _additional {
        answer {
          hasAnswer
          certainty
          property
          result
          startPosition
          endPosition
        }
      }
    }
  }
}
```

```
{
  Get{
    Tweet(
      nearText: {
        concepts: ["forest fires"],
        certainty: 0.7,
      }
    ){
      rawText
      _additional {
        certainty
      }
    }
  }
}
```