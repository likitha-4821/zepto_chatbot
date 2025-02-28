import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Page configuration
st.set_page_config(page_title="Zepto Chatbot", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stChatMessage.user {
            background-color: #0078D7;
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin: 5px;
        }
        .stChatMessage.assistant {
            background-color: #34A853;
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin: 5px;
        }
        .stChatInput {
            border: 2px solid #0078D7;
            border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load dataset
csv_url = "zepto.csv"
try:
    df = pd.read_csv(csv_url)
except Exception as e:
    st.error(f"Failed to load the CSV file. Error: {e}")
    st.stop()

df = df.fillna("")
df['Question'] = df['Question'].str.lower()
df['Answer'] = df['Answer'].str.lower()

# Vectorize questions
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df['Question'])

# Configure Gemini AI
API_KEY = "AIzaSyBKkT0fcb08bmMQBLSu7KU5q8bTNgMjhPI"  
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def find_closest_question(user_query, vectorizer, question_vectors, df):
    query_vector = vectorizer.transform([user_query.lower()])
    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    best_match_index = similarities.argmax()
    best_match_score = similarities[best_match_index]
    if best_match_score > 0.3: 
        return df.iloc[best_match_index]['Answer']
    else:
        return None

# Chatbot title with color
st.markdown("""<h1 style='text-align: center; color: #0078D7;'>Zepto Chatbot</h1>""", unsafe_allow_html=True)
st.write("Welcome to the Zepto Chatbot! Ask me anything about zepto.")

# Display previous chat messages
for message in st.session_state.messages:
    role_class = "user" if message["role"] == "user" else "assistant"
    st.markdown(f'<div class="stChatMessage {role_class}">{message["content"]}</div>', unsafe_allow_html=True)

# User input
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="stChatMessage user">{prompt}</div>', unsafe_allow_html=True)
    
    closest_answer = find_closest_question(prompt, vectorizer, question_vectors, df)
    if closest_answer:
        st.session_state.messages.append({"role": "assistant", "content": closest_answer})
        st.markdown(f'<div class="stChatMessage assistant">{closest_answer}</div>', unsafe_allow_html=True)
    else:
        try:
            response = model.generate_content(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            st.markdown(f'<div class="stChatMessage assistant">{response.text}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Sorry, I couldn't generate a response. Error: {e}")
