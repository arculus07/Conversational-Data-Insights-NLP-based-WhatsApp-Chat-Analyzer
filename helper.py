import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import emoji

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = sum(df['message'].str.split().apply(len))
    media_count = df[df['message'] == '<Media omitted>'].shape[0]
    link_count = df['message'].str.contains('http').sum()

    return num_messages, words, media_count, link_count


def fetch_busy(df):
    user_counts = df['user'].value_counts().head()
    user_percent = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index()
    user_percent.columns = ['user', 'percentage']
    return user_counts, user_percent

def create_wordC(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'].str.strip() != '']  # Remove empty messages

    words = " ".join(temp['message']).strip()  # Combine all messages into a single string
    if len(words) < 10:  # If no words exist, provide a default placeholder
        words = "never talked"  # <-- Placeholder text to avoid errors

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='black').generate(words)
    return wc


def mc_word(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = df[df['user'] != '<media omitted>\n']
    temp = temp[temp['message'] != '']
    words = []
    for message in temp['message']:
        words.extend(message.split())
    if len(words) == 0:
        words = ["you","should","chat","more"]
    r_df = pd.DataFrame(Counter(words).most_common(20))
    return r_df


def emoji_help(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df


def monthly_time(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)
    return timeline


def daily_time(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    daily_timeline = df.groupby('timeDate').count()['message'].reset_index()
    return daily_timeline


def day_count(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()


def month_count(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()


def classification_distribution(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    class_counts = df['classification'].value_counts()
    total_messages = class_counts.sum()
    class_percentages = (class_counts / total_messages * 100).round(2)

    fig, ax = plt.subplots()
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax, palette=['red', 'gray', 'green'])

    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 2, f"{class_percentages.iloc[i]}%", ha='center', fontsize=12, fontweight='bold')

    ax.set_title(f'Message Classification Distribution - {selected_user}')
    ax.set_ylabel('Count')
    ax.set_xlabel('Category')

    return fig, class_counts, class_percentages

with open("stop_hinglish.txt", "r", encoding="utf-8") as f:
    hinglish_stopwords = f.read().splitlines()
def group_users(df):
    vectorizer = TfidfVectorizer(stop_words=hinglish_stopwords)

    # df = df[df['user'] != 'group_notification']
    user_messages = df.groupby('user')['message'].apply(lambda x: ' '.join(x)).reset_index()

    x = vectorizer.fit_transform(user_messages['message'])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

    user_messages['group'] = kmeans.fit_predict(x)

    return user_messages


def categorize_users(df):
    # Count messages per classification for each user
    df = df[df['user'] != 'group_notification']
    user_classification = df.groupby(['user', 'classification']).size().unstack(fill_value=0)

    # Ensure all categories exist in columns
    for category in ['Neutral', 'Friendly', 'Toxic']:
        if category not in user_classification.columns:
            user_classification[category] = 0

    # Calculate total messages per user
    user_classification['total_messages'] = user_classification[['Neutral', 'Friendly', 'Toxic']].sum(axis=1)

    # Calculate percentage of each classification
    user_classification['Neutral_ratio'] = user_classification['Neutral'] / user_classification['total_messages']
    user_classification['Friendly_ratio'] = user_classification['Friendly'] / user_classification['total_messages']
    user_classification['Toxic_ratio'] = user_classification['Toxic'] / user_classification['total_messages']

    # Set a threshold to avoid neutral dominance
    def assign_category(row):
        if row['Toxic_ratio'] >= 0.1:  # At least 40% toxic messages
            return 'Toxic'
        elif row['Friendly_ratio'] >= 0.05:  # At least 40% friendly messages
            return 'Friendly'
        else:
            return 'Neutral'  # Default if no category is dominant enough

    # Apply function to determine the dominant category
    user_classification['dominant_category'] = user_classification.apply(assign_category, axis=1)

    # Categorize users based on dominant classification
    neutral_users = user_classification[user_classification['dominant_category'] == 'Neutral'].index.tolist()
    friendly_users = user_classification[user_classification['dominant_category'] == 'Friendly'].index.tolist()
    toxic_users = user_classification[user_classification['dominant_category'] == 'Toxic'].index.tolist()

    return neutral_users, friendly_users, toxic_users


from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import cosine_similarity


def find_most_similar_user(df, selected_user):
    vectorizer = TfidfVectorizer(stop_words=hinglish_stopwords)

    # Combine all messages per user
    user_messages = df.groupby('user')['message'].apply(lambda x: ' '.join(x)).reset_index()

    if selected_user not in user_messages['user'].values:
        return None  # If user not found in dataset

    x = vectorizer.fit_transform(user_messages['message'])

    cosine_sim = cosine_similarity(x)
    user_index = user_messages[user_messages['user'] == selected_user].index[0]

    # Get all users sorted by similarity score
    similar_scores = list(enumerate(cosine_sim[user_index]))
    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)

    for index, score in similar_scores:
        if index != user_index:  # Ensure we don't pick the selected user
            return user_messages.iloc[index]['user']  # Return the most similar user

    # If no one else is found, return any other user (fallback)
    other_users = user_messages[user_messages['user'] != selected_user]
    if not other_users.empty:
        return other_users.iloc[0]['user']

    return None  # If absolutely no one else is there
# If no match found, return the first user in the dataset

from datetime import datetime, timedelta


def filter_by_date(df, date_option, start_date=None, end_date=None):
    """
    Filters the DataFrame based on the selected date range.
    Ensures at least three unique users exist for K-Means clustering.
    """
    if "date" not in df.columns:
        return df  # Return original DataFrame if no date column found

    df['date'] = pd.to_datetime(df['date'])  # Ensure date column is in datetime format
    today = datetime.today()

    if date_option == "Last Week":
        start_date = today - timedelta(days=7)
    elif date_option == "Last Month":
        start_date = today - timedelta(days=30)
    elif date_option == "Last 6 Months":
        start_date = today - timedelta(days=180)
    elif date_option == "Last Year":
        start_date = today - timedelta(days=365)

    # Apply date filtering
    if date_option != "All Time" and date_option != "Custom Input":
        df = df[df['date'] >= start_date]

    if date_option == "Custom Input" and start_date and end_date:
        df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

    # Ensure at least three unique users exist for K-Means
    unique_users = df['user'].unique().tolist()

    if len(unique_users) < 3:
        new_row = pd.DataFrame([{'user': 'group_notification', 'message': 'System Message', 'date': df['date'].min()}])
        df = pd.concat([df, new_row], ignore_index=True)
    return df





