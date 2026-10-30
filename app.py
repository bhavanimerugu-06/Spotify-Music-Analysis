
import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Spotify Music Analysis",
    page_icon="🎵",
    layout="wide"
)

# Load and clean dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/spotify_tracks.csv")

    # Remove unnecessary index column
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove records with missing important values
    df = df.dropna(subset=["track_name", "artists", "track_genre"])

    # Convert numeric columns
    numeric_cols = [
        "popularity", "danceability", "energy",
        "tempo", "valence", "acousticness",
        "loudness", "speechiness", "instrumentalness"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["popularity", "energy", "danceability"])

    return df


df = load_data()

# Dashboard title
st.title("🎵 Spotify Music Analysis Dashboard")
st.write("Explore songs, artists, genres and audio features using data analytics.")

st.divider()

# Sidebar filters
st.sidebar.header("Filter Songs")

genres = ["All Genres"] + sorted(df["track_genre"].unique().tolist())

selected_genre = st.sidebar.selectbox(
    "Select Genre",
    genres
)

min_popularity = st.sidebar.slider(
    "Minimum Song Popularity",
    0, 100, 0
)

# Apply filters
filtered_df = df[df["popularity"] >= min_popularity]

if selected_genre != "All Genres":
    filtered_df = filtered_df[
        filtered_df["track_genre"] == selected_genre
    ]

# Dashboard
if filtered_df.empty:
    st.warning("No songs found. Please change your filters.")

else:
    # Key performance indicators
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Songs", f"{len(filtered_df):,}")
    col2.metric("Total Artists", f"{filtered_df['artists'].nunique():,}")
    col3.metric("Total Genres", f"{filtered_df['track_genre'].nunique():,}")
    col4.metric(
        "Average Popularity",
        f"{filtered_df['popularity'].mean():.1f}/100"
    )

    st.divider()

    # Top 10 popular songs
    st.subheader("🏆 Top 10 Popular Songs")

    top_songs = filtered_df.nlargest(10, "popularity").copy()
    top_songs["Song"] = (
        top_songs["track_name"] + " - " + top_songs["artists"]
    )

    fig1 = px.bar(
        top_songs.sort_values("popularity"),
        x="popularity",
        y="Song",
        orientation="h",
        color="popularity",
        title="Most Popular Songs",
        labels={"popularity": "Popularity", "Song": "Track"}
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Genre analysis
    st.subheader("🎼 Song Distribution by Genre")

    genre_counts = (
        filtered_df["track_genre"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    genre_counts.columns = ["Genre", "Number of Songs"]

    fig2 = px.bar(
        genre_counts,
        x="Genre",
        y="Number of Songs",
        color="Number of Songs",
        title="Top 10 Genres by Number of Songs"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Audio feature analysis
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("⚡ Energy Distribution")

        fig3 = px.histogram(
            filtered_df,
            x="energy",
            nbins=30,
            title="Distribution of Song Energy"
        )

        st.plotly_chart(fig3, use_container_width=True)

    with col6:
        st.subheader("💃 Danceability vs Popularity")

        sample_df = filtered_df.sample(
            n=min(3000, len(filtered_df)),
            random_state=42
        )

        fig4 = px.scatter(
            sample_df,
            x="danceability",
            y="popularity",
            color="energy",
            title="Danceability and Song Popularity",
            opacity=0.6
        )

        st.plotly_chart(fig4, use_container_width=True)

    # Correlation analysis
    st.subheader("📊 Audio Features Correlation")

    features = [
        "popularity", "danceability", "energy",
        "valence", "tempo", "acousticness"
    ]

    correlation = filtered_df[features].corr()

    fig5 = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap of Audio Features",
        zmin=-1,
        zmax=1
    )

    st.plotly_chart(fig5, use_container_width=True)

    # Key insights
    st.subheader("💡 Key Insights")

    most_popular = filtered_df.loc[
        filtered_df["popularity"].idxmax()
    ]

    st.write(
        f"1. Most popular song: **{most_popular['track_name']}** "
        f"by **{most_popular['artists']}**."
    )

    st.write(
        f"2. Average song energy is "
        f"**{filtered_df['energy'].mean():.2f}**."
    )

    st.write(
        f"3. Average danceability is "
        f"**{filtered_df['danceability'].mean():.2f}**."
    )

    st.write(
        f"4. The dataset contains "
        f"**{filtered_df['track_genre'].nunique()} genres** "
        f"under the selected filter."
    )

    # Dataset preview
    with st.expander("View Dataset"):
        st.dataframe(filtered_df.head(100), use_container_width=True)

st.divider()
st.caption("Spotify Music Analysis | Data Analytics College Project")
