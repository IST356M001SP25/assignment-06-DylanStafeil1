import streamlit as st
import pandas as pd
import requests
import json 
if __name__ == "__main__":
    import sys
    sys.path.append('code')
    from apicalls import get_google_place_details, get_azure_sentiment, get_azure_named_entity_recognition
else:
    from code.apicalls import get_google_place_details, get_azure_sentiment, get_azure_named_entity_recognition

PLACE_IDS_SOURCE_FILE = "cache/place_ids.csv"
CACHE_REVIEWS_FILE = "cache/reviews.csv"
CACHE_SENTIMENT_FILE = "cache/reviews_sentiment_by_sentence.csv"
CACHE_ENTITIES_FILE = "cache/reviews_sentiment_by_sentence_with_entities.csv"


def reviews_step(place_ids: str|pd.DataFrame) -> pd.DataFrame:
    '''
      1. place_ids --> reviews_step --> reviews: place_id, name (of place), author_name, rating, text 
    '''
    # Check if the input is a string (filename) or a DataFrame
    if isinstance(place_ids, str):
        # Read the CSV file into a DataFrame
        place_ids_df = pd.read_csv(place_ids)
    elif isinstance(place_ids, pd.DataFrame):
        # Use the provided DataFrame directly
        place_ids_df = place_ids
    else:
        raise ValueError("Input must be a filename or a DataFrame.")
    
    # Initialize an empty list to store the reviews
    reviews_list = []

    # Iterate through each place_id in the DataFrame
    for index, row in place_ids_df.iterrows():
        # Extract the place_id
        place_id = row['Google Place ID'] 
        # Call the Google Places API to get the place details and reviews
        response = get_google_place_details(place_id)
        # Check if 'result' key exists in the response
        if 'result' in response:
            # Append the reviews to the list
            reviews_list.append(response['result'])

    # Normalize the JSON data to create a DataFrame
    reviews_df = pd.json_normalize(reviews_list, record_path=['reviews'], meta=['place_id', 'name'])

    # Filter the DataFrame to keep only the relevant columns
    reviews_df = reviews_df[['place_id', 'name', 'author_name', 'rating', 'text']]
    
    # Write the DataFrame to a CSV file
    reviews_df.to_csv(CACHE_REVIEWS_FILE, index=False)

    return reviews_df

def sentiment_step(reviews: str|pd.DataFrame) -> pd.DataFrame:
    '''
      2. reviews --> sentiment_step --> review_sentiment_by_sentence
    '''
    # Check if the input is a string (filename) or a DataFrame
    if isinstance(reviews, str):
        # Read the CSV file into a DataFrame
        reviews_df = pd.read_csv(reviews)
    elif isinstance(reviews, pd.DataFrame):
        # Use the provided DataFrame directly
        reviews_df = reviews
    else:
        raise ValueError("Input must be a filename or a DataFrame.")
    
    # Initialize an empty list to store the sentiment results
    sentiment_results = []

    # Iterate through each review in the DataFrame
    for index, row in reviews_df.iterrows():
        # Extract the text of the review
        text = row['text']
        # Call the Azure Sentiment API to get the sentiment of the text
        response = get_azure_sentiment(text)
        # Check if 'results' key exists in the response
        if 'results' in response:
            # Extract the sentiment results and add place_id and name
            sentiment_result = response['results']['documents'][0]
            sentiment_result['place_id'] = row['place_id']
            sentiment_result['name'] = row['name']
            sentiment_result['author_name'] = row['author_name']
            sentiment_result['rating'] = row['rating']
            # Append the result to the list
            sentiment_results.append(sentiment_result)

    # Normalize the JSON data to create a DataFrame
    sentiment_df = pd.json_normalize(sentiment_results)

    # Transform the DataFrame to be at the sentences level
    sentiment_df = sentiment_df.explode('sentences')

    # Normalize the sentences column to create a new DataFrame
    sentences_df = pd.json_normalize(sentiment_df['sentences'])

    # Concatenate the sentences DataFrame with the original DataFrame
    sentiment_df = pd.concat([
        sentiment_df.drop(columns=['sentences']).reset_index(drop=True),
        sentences_df.reset_index(drop=True)
    ], axis=1)

    # Rename the columns
    sentiment_df.rename(columns={
        'text': 'sentence_text',
        'sentiment': 'sentence_sentiment',
        'confidenceScores.positive': 'confidenceScores_positive',
        'confidenceScores.neutral': 'confidenceScores_neutral',
        'confidenceScores.negative': 'confidenceScores_negative'
    }, inplace=True)

    # Filter the DataFrame to keep only the relevant columns
    sentiment_df = sentiment_df[['place_id', 'name', 'author_name', 'rating', 'sentence_text', 'sentence_sentiment', 
                             'confidenceScores_positive', 'confidenceScores_neutral', 
                             'confidenceScores_negative']]
    
    # Write the DataFrame to a CSV file
    sentiment_df.to_csv(CACHE_SENTIMENT_FILE, index=False)

    return sentiment_df



def entity_extraction_step(sentiment: str|pd.DataFrame) -> pd.DataFrame:
    '''
      3. review_sentiment_by_sentence --> entity_extraction_step --> review_sentiment_entities_by_sentence
    '''
    # Check if the input is a string (filename) or a DataFrame
    if isinstance(sentiment, str):
        # Read the CSV file into a DataFrame
        sentiment_df = pd.read_csv(sentiment)
    elif isinstance(sentiment, pd.DataFrame):
        # Use the provided DataFrame directly
        sentiment_df = sentiment
    else:
        raise ValueError("Input must be a filename or a DataFrame.")
    
    # Initialize an empty list to store the entity extraction results
    entity_results = []

    # Iterate through each sentence in the DataFrame
    for index, row in sentiment_df.iterrows():
        # Extract the text of the sentence
        text = row['sentence_text']
        # Call the Azure Named Entity Recognition API to get the entities in the text
        response = get_azure_named_entity_recognition(text)
        # Check if 'results' key exists in the response
        if 'results' in response:
            # Extract the entity results and add place_id, name, author_name, rating, sentence_text, sentence_sentiment
            entity_result = response['results']['documents'][0]
            for col in sentiment_df.columns:
                entity_result[col] = row[col]    
            # Append the result to the list
            entity_results.append(entity_result)

    # Normalize the JSON data to create a DataFrame
    entities_df = pd.json_normalize(entity_results, record_path=['entities'], meta=list(sentiment_df.columns))

    # Rename the columns
    entities_df.rename(columns={
        'text': 'entity_text',
        'category': 'entity_category',
        'subcategory': 'entity_subcategory',
        'confidenceScore': 'confidenceScores_entity'
    }, inplace=True)

    # Filter the DataFrame to keep only the relevant columns
    entities_df = entities_df[['place_id', 'name', 'author_name', 'rating', 'sentence_text', 
                                'sentence_sentiment', 'confidenceScores_positive', 'confidenceScores_neutral', 
                                'confidenceScores_negative', 'entity_text', 'entity_category', 
                                'entity_subcategory', 'confidenceScores_entity']]
    
    # Write the DataFrame to a CSV file
    entities_df.to_csv(CACHE_ENTITIES_FILE, index=False)

    return entities_df


if __name__ == '__main__':
    # helpful for debugging as you can view your dataframes and json outputs
    import streamlit as st 
    reviews_step(PLACE_IDS_SOURCE_FILE)
    sentiment_step(CACHE_REVIEWS_FILE)
    entities_df = entity_extraction_step(CACHE_SENTIMENT_FILE)
    st.write(entities_df)