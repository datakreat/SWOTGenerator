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
from qdrant_client import QdrantClient
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_groq import ChatGroq
import requests
# Initialize the Reddit client
load_dotenv()
reddit = praw.Reddit(
    client_id=os.getenv('reddit_client_id'),
    client_secret=os.getenv('reddit_client_secret'),
    user_agent=os.getenv('reddit_user_agent')
)

embeddings = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key="HgZpQb9mGtu3BmkG9AtMQVhT9kRW9VHSqSXk2kqP")
# llm = ChatGroq(temperature=0.5, groq_api_key="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", model_name="llama3-70b-8192")

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url="https://f4bf0d03-f13c-43b6-87cc-32935393ab68.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="r8CbzHB7iEGgX5ISK885jtMkaHCLJYAceYvjx3V-OEtKHqn1xnzHvQ"
)

# Initialize collections
collections = {
    "innovations": Qdrant(
        client=qdrant_client,
        collection_name="biomimicry-innovations",
        embeddings=embeddings
    ),
    "strategies": Qdrant(
        client=qdrant_client,
        collection_name="biomimicry-strategies",
        embeddings=embeddings
    ),
    "journals": Qdrant(
        client=qdrant_client,
        collection_name="journals",
        embeddings=embeddings
    ),
    "patents": Qdrant(
        client=qdrant_client,
        collection_name="patents",
        embeddings=embeddings
    )
}

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

def display_patents(found_docs):
    st.subheader("Related Patents")
    for idx, (doc, score) in enumerate(found_docs[:5]):  # Display top 5 patents
        st.markdown(f"### Patent {idx + 1}")
        st.write(f"**Title:** {doc.metadata['patent_title']}")
        st.write(f"**Abstract:** {doc.page_content}")
        st.write(f"**Score:** {score:.4f}")
        st.write("---")

# Modify your get_similar_patents function to use Qdrant
def get_similar_patents(topic):
    query = f"Domain: {topic}"
    found_docs = collections['patents'].similarity_search_with_score(query, k=10)
    return found_docs

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

def format_patents_info(patents_metadata):
    formatted_info = ""
    for i, patent in enumerate(patents_metadata[:5], 1):  # Limit to top 5 patents
        formatted_info += f"Patent {i}:\n"
        formatted_info += f"Title: {patent.get('patent_title', 'N/A')}\n"
        formatted_info += f"Abstract: {patent.get('patent_abstract', 'N/A')}\n"
        formatted_info += f"Summary: {patent.get('summary_text', 'N/A')}\n\n"
    return formatted_info

def strength(topic, bg):
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
    **Strengths of topic:**
    1. [Strength 1]: Brief explanation
    2. [Strength 2]: Brief explanation
    3. [Strength 3]: Brief explanation
    4. [Strength 4]: Brief explanation
    5. [Strength 5]: Brief explanation (if applicable)

    Given Topic:
    {topic}

    Here is some relevant Background information:
    {bg}
    '''
    return llm.generate_response(prompt)

def weakness(topic, sentiment):
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
    **Weaknesses of topic:**
    1. [Weakness 1]: Brief explanation
    2. [Weakness 2]: Brief explanation
    3. [Weakness 3]: Brief explanation
    4. [Weakness 4]: Brief explanation
    5. [Weakness 5]: Brief explanation (if applicable)

    Topic:
    {topic}

    Here are the some customer sentients from reddit:
    {sentiment}
    '''
    return llm.generate_response(prompt)

def oppurtunity(topic, tech, comp):
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
    **Opportunities for topic:**
    1. [Opportunity 1]: Brief explanation
    2. [Opportunity 2]: Brief explanation
    3. [Opportunity 3]: Brief explanation
    4. [Opportunity 4]: Brief explanation
    5. [Opportunity 5]: Brief explanation (if applicable)

    Topic:
    {topic}

    Here are some patents that might help you:
    {format_patents_info(tech)}

    Also some data on competitiors on given topic:
    {comp}
    '''

    return llm.generate_response(prompt)

def threats(topic, competition):
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
    **Threats to topic:**
    1. [Threat 1]: Brief explanation
    2. [Threat 2]: Brief explanation
    3. [Threat 3]: Brief explanation
    4. [Threat 4]: Brief explanation
    5. [Threat 5]: Brief explanation (if applicable)
    6. [Threat 6]: Brief explanation (if applicable)

    Here is some competitior data for threat analysis:
    {competition}

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
        # Generate background information
        background_info = bg_info(topic)
        st.write(background_info)

        # Analyze Reddit sentiment
        sentiment_score, reddit_posts = reddit_sentiment(topic)
        st.subheader("Reddit Sentiment Analysis")
        st.write(f'#### Sentiment Score: {sentiment_score:.2f}')
        st.write("Sample Reddit Posts:")
        for post in reddit_posts[:3]:  # Display first 3 posts
            st.write(f"- {post[:200]}...")  # Truncate long posts

        # Get similar patents using Qdrant
        similar_patents = get_similar_patents(topic)
        display_patents(similar_patents)

        # Find competition
        competition_info = find_competition(topic)
        st.subheader("Competitive Landscape")
        st.write(competition_info)

        # Generate SWOT analysis
        st.write("# SWOT Analysis")
        
        st.write("## Strengths:")
        st.write(strength(topic, background_info))

        st.write("## Weaknesses:")
        st.write(weakness(topic, reddit_posts))

        st.write("## Opportunities:")
        # Prepare all patents' metadata
        all_patents_metadata = [doc[0].metadata for doc in similar_patents] if similar_patents else []
        st.write(oppurtunity(topic, all_patents_metadata, competition_info))

        st.write("## Threats:")
        st.write(threats(topic, competition_info))
    else:
        st.error("Please enter a topic.")