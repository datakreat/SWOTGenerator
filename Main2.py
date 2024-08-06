import streamlit as st
import llm
import praw
from textblob import TextBlob
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the Reddit client
load_dotenv()
reddit = praw.Reddit(
    client_id=os.getenv('reddit_client_id'),
    client_secret=os.getenv('reddit_client_secret'),
    user_agent=os.getenv('reddit_user_agent')
)

def get_reddit_posts(query, limit=10):
    posts = reddit.subreddit('all').search(query, limit=limit)
    return [post.title + " " + post.selftext for post in posts]

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Returns a value between -1 and 1

def reddit_sentiment(patent_title):
    # print(f"Searching Reddit for posts about: {patent_title}")
    posts = get_reddit_posts(patent_title)

    if not posts:
        print("No posts found.")
        return

    sentiments = []
    for idx, post in enumerate(posts):
        sentiment = analyze_sentiment(post)
        sentiments.append(sentiment)
        # print(f"Post {idx+1}:")
        # print(post)
        # print(f"Sentiment: {sentiment:.2f}\n")

    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    return avg_sentiment, posts

def load_data():
    df = pd.read_csv('assignee_patents_with_all_citations_and_grants.csv')
    return df

def load_similarity_matrix():
    with open('similarity_matrix.pkl', 'rb') as file:
        similarity_matrix = pickle.load(file)
    return similarity_matrix

def bg_info(topic):
    prompt = f'''
    You are Kreat.ai. Your job is to to provide background info in two lines on the following topic:

    {topic}

    Provide relevant info only.
    '''
    answer = llm.generate_response(prompt)
    return '### Background Information:\n' + answer

def get_similar_patents(description, df, similarity_matrix):
    # Create a TF-IDF vector for the input description
    vectorizer = TfidfVectorizer()
    corpus = df['combined_text'].tolist() + [description]
    vectors = vectorizer.fit_transform(corpus)
    
    # Compute similarity between description vector and all patent vectors
    description_vector = vectors[-1]
    patent_vectors = vectors[:-1]
    
    similarity_scores = cosine_similarity(description_vector, patent_vectors).flatten()
    
    # Get indices of the top patents based on similarity scores
    top_indices = np.argsort(similarity_scores)[::-1]
    
    # Retrieve top patents
    top_patents = df.iloc[top_indices].copy()
    top_patents['similarity_score'] = similarity_scores[top_indices]
    
    # Calculate total citations
    top_patents['total_citations'] = top_patents['foreign_citation_count'] + top_patents['us_citation_count']
    
    # Rank patents by similarity_score and total_citations first, then by novelty_score
    novelty_order = {'High': 1, 'Medium': 2, 'Low': 3}
    top_patents['novelty_rank'] = top_patents['novelty_score'].map(novelty_order)
    
    # Sort by similarity_score and total_citations first, then by novelty_rank
    top_patents = top_patents.sort_values(by=['similarity_score', 'total_citations', 'novelty_rank'], ascending=[False, False, True])
    
    return top_patents

def find_competition(topic):
    prompt = f'''
    Provide a detailed analysis of the key competitors in the industry/market related to {topic}. Include the following aspects for each competitor:

    Format:
    #### Company Name: The names of the competitor.
    #### Strengths: What are the key strengths of this competitor in relation to [specific topic]?
    #### Weaknesses: What are the major weaknesses or challenges faced by this competitor?
    #### Market Position: How is this competitor positioned in the market (e.g., market share, target segments)?
    #### Recent Developments: Any recent developments or changes related to this competitor (e.g., new products, mergers, strategic moves)?
    #### Please provide information on the top competitors in this field.
    '''
    return llm.generate_response(prompt)

def strength(topic):
    prompt = f'''
    You are an advanced AI model providing SWOT analysis for an innovator. Your task is to identify and list concise strengths of a given topic. 

    Guidelines:
    1. Unique advantages or capabilities
    2. Superior features compared to alternatives
    3. Proprietary technologies or patents
    4. Strong brand recognition or market position
    5. Exceptional team or expertise
    6. Cost efficiencies or high profitability

    Ensure each strength is:
    - Specific and measurable where possible
    - Relevant to the current market
    - Truly a strength, not just a feature
    
    Format your response as follows:
    **Strengths of {topic}:**
    1. [Strength 1]: Brief explanation
    2. [Strength 2]: Brief explanation
    3. [Strength 3]: Brief explanation
    4. [Strength 4]: Brief explanation
    5. [Strength 5]: Brief explanation (if applicable)

    Topic:
    {topic}
    '''
    return llm.generate_response(prompt)


def weakness(topic):
    prompt = f'''
    You are an advanced AI model providing SWOT analysis for an innovator. Your task is to identify and list concise weaknesses of a given topic.

    Guidelines:
    1. Technological limitations or drawbacks
    2. Resource constraints (financial, human, or material)
    3. Gaps in product lineup or market coverage
    4. Lack of certain expertise or capabilities
    5. Dependency on external factors
    6. High costs or low profitability areas

    Ensure each weakness is:
    - Honest and realistic
    - Actionable or potentially addressable
    - Backed by evidence or logical reasoning

    Format your response as follows:
    **Weaknesses of {topic}:**
    1. [Weakness 1]: Brief explanation
    2. [Weakness 2]: Brief explanation
    3. [Weakness 3]: Brief explanation
    4. [Weakness 4]: Brief explanation
    5. [Weakness 5]: Brief explanation (if applicable)

    Topic:
    {topic}
    '''
    return llm.generate_response(prompt)

def opportunity(topic):
    prompt = f'''
    You are an advanced AI model providing SWOT analysis for an innovator. Your task is to generate and list possible opportunities related to a given topic.

    Guidelines:
    1. Emerging market trends or consumer needs
    2. Technological advancements that could be leveraged
    3. Potential for expansion into new markets or segments
    4. Possible strategic partnerships or acquisitions
    5. Changes in regulations or policies that could be beneficial
    6. Weaknesses of competitors that could be exploited

    Ensure each opportunity is:
    - Forward-looking and growth-oriented
    - Aligned with the topic's strengths where possible
    - Realistic and potentially achievable

    Format your response as follows:
    **Opportunities for {topic}:**
    1. [Opportunity 1]: Brief explanation
    2. [Opportunity 2]: Brief explanation
    3. [Opportunity 3]: Brief explanation
    4. [Opportunity 4]: Brief explanation
    5. [Opportunity 5]: Brief explanation (if applicable)

    Topic:
    {topic}
    '''
    return llm.generate_response(prompt)

def threats(topic):
    prompt = f'''
    You are an advanced AI model providing SWOT analysis for an innovator. Your task is to identify and list possible threats related to a given topic.

    Guidelines, provide 4-6 key threats for the given topic. Consider:
    1. Emerging competitors or disruptive technologies
    2. Changing market dynamics or consumer preferences
    3. Potential regulatory challenges or policy changes
    4. Economic factors that could impact the industry
    5. Supply chain vulnerabilities or resource scarcity
    6. Cybersecurity risks or data privacy concerns

    Ensure each threat is:
    - Relevant to the current or near-future landscape
    - Specific and backed by trend analysis where possible
    - Potentially impactful on the topic's success or viability

    Format your response as follows:
    **Threats to {topic}:**
    1. [Threat 1]: Brief explanation
    2. [Threat 2]: Brief explanation
    3. [Threat 3]: Brief explanation
    4. [Threat 4]: Brief explanation
    5. [Threat 5]: Brief explanation (if applicable)
    6. [Threat 6]: Brief explanation (if applicable)

    Topic:
    {topic}
    '''
    return llm.generate_response(prompt)

# Streamlit app
st.title("SWOT Analysis Generator")

st.header("Generate SWOT Analysis")

# Input field for the topic
topic = st.text_input("Enter a topic:", "")

# Generate SWOT analysis when the button is pressed
if st.button("Generate SWOT Analysis"):
    if topic:
        # Load data and similarity matrix
        # df = load_data()
        # similarity_matrix = load_similarity_matrix()
        # if 'combined_text' not in df.columns:
        #     df['combined_text'] = df['patent_title'] + " " + df['patent_abstract']

        # # Generate background information
        # background_info = bg_info(topic)
        # st.write(background_info)

        # # Analyze Reddit sentiment
        # sentiment_score, reddit_posts = reddit_sentiment(topic)
        # st.subheader("Reddit Sentiment Analysis")
        # st.write(f'#### Sentiment Score: {sentiment_score}')
        # st.write("Sample Reddit Posts:")
        # for post in reddit_posts[:3]:  # Display first 3 posts
        #     st.write(f"- {post[:200]}...")  # Truncate long posts

        # # Get similar patents
        # top_patents = get_similar_patents(topic, df, similarity_matrix)
        # best_patent = top_patents.iloc[0] if not top_patents.empty else None
        # st.subheader("Best Patent:")
        # if best_patent is not None:
        #     st.write(f"**Title:** {best_patent['patent_title']}")
        #     st.write(f"**Abstract:** {best_patent['patent_abstract']}")
        #     st.write(f"**Novelty Score:** {best_patent['novelty_score']}")
        #     st.write(f"**Total Citations:** {best_patent['total_citations']}")
        #     st.write(f"**Similarity Score:** {best_patent['similarity_score']:.2f}")
        # else:
        #     st.write('No patents found')

        # # Find competition
        # competition_info = find_competition(topic)
        # st.subheader("Competitive Landscape")
        # st.write(competition_info)

        # # Generate SWOT analysis
        # st.write("# SWOT Analysis")
        
        st.write("## Strengths:")
        st.write(strength(topic))

        st.write("## Weaknesses:")
        st.write(weakness(topic))

        st.write("## Opportunities:")
        st.write(opportunity(topic))

        st.write("## Threats:")
        st.write(threats(topic))

    else:
        st.error("Please enter a topic.")