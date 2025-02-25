from datetime import datetime
import pandas as pd
import re
import string
from textblob import TextBlob
import re
import emoji

# Load abusive words
with open("abusive_words.txt", "r", encoding="utf-8") as f:
    abusive_words = set(f.read().splitlines())

# Load stop words
with open('stop_hinglish.txt', 'r', encoding="utf-8") as f:
    stop_words = set(word.strip().lower() for word in f.readlines())


def preprocess(data):
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})\s?(AM|PM)\s-\s'

    def convert_to_24h(match):
        date_part = match.group(1)
        time_part = match.group(2)
        am_pm = match.group(3)
        time_24h = datetime.strptime(f"{time_part} {am_pm}", "%I:%M %p").strftime("%H:%M")
        return f"{date_part}, {time_24h} - "

    data = re.sub(pattern, convert_to_24h, data)

    with open("converted_file.txt", "w", encoding="utf-8") as f:
        f.write(data)

    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{2}:\d{2}\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    df['message_date'] = df['message_date'].str.replace(r'\s-\s$', '', regex=True)
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %H:%M', errors='coerce')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['timeDate'] = df['date'].dt.date
    df['day_name'] = df['date'].dt.day_name()

    # Apply text preprocessing and classification
    df['message'] = df['message'].apply(preprocess_text)
    df['classification'] = df['message'].apply(classify_message)

    return df



def preprocess_text(text):
    if not isinstance(text, str):
        return ""  # Handle None/null values safely

    text = text.lower()

    # Remove unwanted phrases
    unwanted_phrases = ["media omitted", "message deleted", "this message was deleted", "null","deleted message","message","deleted"]
    for phrase in unwanted_phrases:
        text = text.replace(phrase, "")

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Keep only letters, numbers, spaces, and emojis
    text = "".join([char for char in text if char.isalnum() or char.isspace() or emoji.is_emoji(char)])

    # Remove stopwords and short words
    text = " ".join([word for word in text.split() if len(word) > 1 and word not in stop_words])

    return text



def classify_message(message):
    words = set(message.lower().split())
    if words & abusive_words:
        return "Toxic"
    polarity = TextBlob(message).sentiment.polarity
    return "Friendly" if polarity > 0 else "Neutral"
