import streamlit as st
from numpy.matlib import empty

import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page config
st.set_page_config(page_title="Chat Analyzer", page_icon="💬", layout="wide")

# Custom Styling
st.markdown("""
    <style>
        .css-1d391kg {background-color: #F8F9FA;}
        .stTitle {color: #3498DB; font-size: 36px; font-weight: bold;}
        .stHeader {color: #2C3E50; font-size: 24px;}
        .metric-container {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);}
        .emoji-header {color: #F39C12;}
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📂 Upload Chat File")
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    st.title("📜 Chat Data")
    st.dataframe(df, use_container_width=True)
    # Sidebar - Date Selection
    st.sidebar.subheader("📅 Select Date Range")
    date_option = st.sidebar.selectbox("Filter Data By:",
                                       ["All Time", "Last Week", "Last Month", "Last 6 Months", "Last Year",
                                        "Custom Input"])

    # If custom input is selected, show date picker
    start_date, end_date = None, None
    if date_option == "Custom Input":
        start_date = st.sidebar.date_input("Start Date")
        end_date = st.sidebar.date_input("End Date")

    # Filter data based on selected date range
    df = helper.filter_by_date(df, date_option, start_date, end_date)

    st.sidebar.subheader("👤 Select User")
    user_list = sorted(df['user'].unique().tolist())
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show Analysis for", user_list)

    if st.sidebar.button("🔍 Show Analysis"):
        st.markdown("<h1 class='stTitle'>📊 Chat Analysis</h1>", unsafe_allow_html=True)

        # Statistics Section
        n_m, words, num_med, num_l = helper.fetch_stats(selected_user, df)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(label="💬 Total Messages", value=n_m)
        with col2:
            st.metric(label="📝 Total Words", value=words)
        with col3:
            st.metric(label="📸 Media Shared", value=num_med)
        with col4:
            st.metric(label="🔗 Links Shared", value=num_l)

        # Busy Users (Only for group chats)
        if selected_user == 'Overall':
            st.subheader("🏆 Most Active Users")
            x, n_df = helper.fetch_busy(df)
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                sns.barplot(x=x.index, y=x.values, ax=ax, palette="Oranges")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            with col2:
                st.dataframe(n_df)

        # Word Cloud
        st.subheader("☁️ Word Cloud")
        df_w = helper.create_wordC(selected_user, df)

        fig, ax = plt.subplots()
        ax.imshow(df_w, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # Most Common Words
        st.subheader("🔤 Most Common Words")
        most_common_df = helper.mc_word(selected_user, df)
        fig, ax = plt.subplots()
        sns.barplot(x=most_common_df[1], y=most_common_df[0], ax=ax, palette="blend:#7AB,#EDA")
        st.pyplot(fig)

        # Emojis Analysis
        st.subheader("😀 Emoji Analysis")
        emoji_df = helper.emoji_help(selected_user, df)

        if emoji_df.empty:
            st.write("No emojis used.")
            col1, col2 = st.columns(2)
            with col2:
                fig, ax = plt.subplots()
                ax.pie([1], labels=["No Emojis"], autopct="%0.2f", colors=["lightgrey"])
                st.pyplot(fig)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(emoji_df.iloc[:5, 1], labels=emoji_df.iloc[:5, 0], autopct="%0.2f",
                       colors=sns.color_palette("Set3"))
                st.pyplot(fig)

        # Message Timeline
        st.subheader("📅 Monthly Timeline")
        timeline = helper.monthly_time(selected_user, df)
        fig, ax = plt.subplots()
        sns.lineplot(x=timeline['time'], y=timeline['message'], ax=ax, color='lime')
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.title("📅 Daily Timeline")
        dd_time = helper.daily_time(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(dd_time['timeDate'], dd_time['message'], color='lime')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        st.title("🕒 Most Active Time")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Day-wise Activity")
            b_day = helper.day_count(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(b_day.index, b_day.values, color='lightcoral')
            plt.xticks(rotation=50)
            st.pyplot(fig)

        with col2:
            st.header("Month-wise Activity")
            m_day = helper.month_count(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(m_day.index, m_day.values, color='lightcoral')
            plt.xticks(rotation=50)
            st.pyplot(fig)


        # Toxic & Friendly Members
        st.markdown(
            """
            <style>
            @keyframes fadeIn {
                0% {opacity: 0; transform: translateY(-10px);}
                100% {opacity: 1; transform: translateY(0);}
            }
            .animated-text {
                animation: fadeIn 2s ease-in-out;
                font-size: 26px;
                font-weight: bold;
                text-align: center;
                color: #FF5733;
                background: linear-gradient(to right, #FFDDC1, #FF5733);
                padding: 10px;
                border-radius: 8px;
                box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.2);
            }
            </style>
            <h2 class="animated-text">😈😇 Most Toxic & Friendly Members</h2>
            """,
            unsafe_allow_html=True
        )

        toxic_counts = df[df['classification'] == 'Toxic']['user'].value_counts()
        friendly_counts = df[df['classification'] == 'Friendly']['user'].value_counts()
        most_toxic = toxic_counts.idxmax() if not toxic_counts.empty else "None"
        most_friendly = friendly_counts.idxmax() if not friendly_counts.empty else "None"

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "<h2 style='color:red; text-align:center; text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);'>🔥 Most Toxic</h2>",
                unsafe_allow_html=True)
            st.markdown(
                f"""
                <h3 style='background: linear-gradient(to right, #FFCCCC, #FF7777); 
                color:red; padding:15px; border-radius:10px; text-align:center; 
                box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.2);'>🚨 {most_toxic}</h3>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                "<h2 style='color:green; text-align:center; text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);'>🌿 Most Friendly</h2>",
                unsafe_allow_html=True)
            st.markdown(
                f"""
                <h3 style='background: linear-gradient(to right, #CCFFCC, #77FF77); 
                color:green; padding:15px; border-radius:10px; text-align:center; 
                box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.2);'>💖 {most_friendly}</h3>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br><br>", unsafe_allow_html=True)


        neutral_users, friendly_users, toxic_users = helper.categorize_users(df)

        st.markdown("""
            <style>
                .user-box {
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 10px;
                    margin: 5px 0;
                    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                    transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
                }
                .user-box:hover {
                    transform: scale(1.05);
                    box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.2);
                }
                .premium-title {
                    text-align: center;
                    font-size: 28px;
                    font-weight: bold;
                    color: #4A90E2;
                    padding: 15px;
                    background: linear-gradient(to right, #f8f9fa, #e0e0e0);
                    border-radius: 10px;
                    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                    margin-bottom: 20px;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("<div class='premium-title'>✨ User Categorization Based on Dominant Message Type</div>",
                    unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h3 style='text-align: center; color: #28a745;'>😊 Neutral Users</h3>", unsafe_allow_html=True)
            for user in neutral_users:
                st.markdown(f"<div class='user-box'>👤 {user}</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<h3 style='text-align: center; color: #007bff;'>❤ Friendly Users</h3>", unsafe_allow_html=True)
            for user in friendly_users:
                st.markdown(f"<div class='user-box'>❤ {user}</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<h3 style='text-align: center; color: #dc3545;'>💀 Toxic Users</h3>", unsafe_allow_html=True)
            for user in toxic_users:
                st.markdown(f"<div class='user-box'>⚠️ {user}</div>", unsafe_allow_html=True)

        # Similar Users
        st.subheader("👥 Similar Users")
        if selected_user == "Overall":
            unique_grouped_df = helper.group_users(df)
            for group_num in sorted(unique_grouped_df['group'].unique()):
                users_in_group = unique_grouped_df[unique_grouped_df['group'] == group_num]['user'].tolist()
                with st.expander(f"🔹 Group {group_num + 1}"):
                    st.write(", ".join(users_in_group))
        else:
            most_similar_user = helper.find_most_similar_user(df, selected_user)
            if most_similar_user:
                st.info(f"🔍 **Most Similar User to {selected_user}:** {most_similar_user}")
            else:
                st.warning("No similar user found due to insufficient data.")


        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]
        st.title(f"{selected_user}, Message Classification Distribution")
        class_counts = df['classification'].value_counts()
        fig, ax = plt.subplots()
        colors = ['#FFC0CB', 'lightblue', 'lime']
        ax.bar(class_counts.index, class_counts.values, color=colors)
        plt.xticks(rotation='vertical')
        for i, v in enumerate(class_counts.values):
            ax.text(i, v + 2, f"{(v / class_counts.sum()) * 100:.2f}%", ha='center')
        st.pyplot(fig)

st.sidebar.title("Navigation")
if st.sidebar.button("About Chat Analyzer"):
    st.markdown(
"""
## About Chat Analyzer
Chat Analyzer is an advanced NLP-based tool designed to analyze WhatsApp chat data, providing deep insights into user interactions. Inspired by Nitish Sir from Campus X, this project has been enhanced with numerous additional features to improve accuracy, usability, and overall functionality.

    ### Key Features:
    - **Toxicity Detection:** Identifies harmful or offensive language, helping users moderate conversations effectively.
    - **User Grouping:** Clusters similar users based on their communication patterns, enabling better understanding of conversational dynamics.
    - **Sentiment Analysis:** Analyzes chat sentiments to gauge overall mood and engagement.
    - **Keyword Extraction:** Highlights key topics of discussion within chat groups.
    - **Statistical Insights:** Provides comprehensive metrics such as message frequency, most active users, and word usage patterns.
    - **Scalability & Performance:** Optimized for handling large chat datasets efficiently.

    ### Technology Stack:
    - **Natural Language Processing (NLP):** Advanced text processing techniques for accurate chat analysis.
    - **Machine Learning & AI:** Implements classification and clustering algorithms for intelligent insights.
    - **Python & Libraries:** Utilizes frameworks like TensorFlow, scikit-learn, and spaCy for robust performance.

    ### Use Cases:
    - **Moderation & Safety:** Helps admins and moderators detect toxic behavior in chat groups.
    - **Social & Behavioral Analysis:** Useful for researchers studying communication trends and online interactions.
    - **Personal Chat Insights:** Allows users to understand their messaging habits and relationships within groups.

    ### Developer Information:
    - Developed by **Ayush Ranjan**, a **3rd-year B.Tech Computer Science** student at **Lovely Professional University**.
    - Passionate about **AI, NLP, and real-world problem-solving through technology**.

    With its combination of **AI-driven analytics** and **user-friendly implementation**, Chat Analyzer stands out as a powerful tool for chat analysis, offering meaningful insights for individuals, communities, and researchers alike.
    
    NOTE:- THE MODEL IS DESIGNED TO WORK ON BOTH 24 HOUR FORMAT CHAT AND 12 HOUR FORMAT(AM\PM) BUT IT IS ADVISED TO USE AM\PM ON THE PRIOR BASIS
    """
)

