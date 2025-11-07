import pandas as pd
import numpy as np

colors = {
    'background': '#0f1a2f',
    'card_bg': '#1e2a45',
    'card_border': '#2d3b55',
    'primary': '#2563eb',
    'secondary': '#06b6d4',
    'success': '#10b981',
    'warning': "#382a12",
    'danger': '#ef4444',
    'text_primary': '#f8fafc',
    'text_secondary': '#94a3b8',
    'accent_gradient': 'linear-gradient(135deg, #2563eb 0%, #06b6d4 100%)'
}

df = pd.read_csv("c:\\Users\\Moving_King\\Documents\\BillBoard\\Billboard Hot 100 Number Ones Database - Data.csv")

##******* Data cleaning and preparation************
drop_cols = [
    'Featured Artists', 'Talent Contestant', 'Sound Effects',
    'Featured in a Then Contemporary Film', 'Featured in a then Contemporary T.V. Show',
    'Featured in a Then Contemporary Play', 'Double A Side'
]
df = df.drop(columns=drop_cols)

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

df['Label'].fillna('Unknown', inplace=True)
df['CDR Genre'].fillna('Unknown', inplace=True)
df['CDR Style'].fillna('Unknown', inplace=True)

df['Year'] = pd.to_datetime(df['Date']).dt.year

# Categorize the loudness into levels (e.g., Quiet, Moderate, Loud).
df['Loudness_level'] = pd.cut(df['Loudness (dB)'], bins=[-60, -20, -10, 0], labels=['Quiet', 'Moderate', 'Loud'])

# energy_level: Categorize energy into low, medium, and high levels.
df['Energy_level'] = pd.cut(df['Energy'], bins=[0, 0.33, 0.66, 1], labels=['Low', 'Medium', 'High'])

# danceability_level: Create categories for danceability (e.g., Low, Medium, High).
df['Danceability_level'] = pd.cut(df['Danceability'], bins=[0, 0.33, 0.66, 1], labels=['Low', 'Medium', 'High'])

# explicit_flag: Convert explicit boolean into a more descriptive text (Explicit, Non-Explicit).
df['Explicit_flag'] = df['Explicit'].replace({1: 'Explicit', 0: 'Non-Explicit'})

# Display all columns and rows contained in the dataset
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Create a comprehensive vocal analysis section
vocal_techniques = [
    'Vocally Based', 'Falsetto Vocal', 'Vocal Introduction', 
    'Free Time Vocal Introduction', 'Multiple Lead Vocalists',
    'Spoken Word', 'Explicit', 'Foreign Language', 'Rap Verse in a Non-Rap Song'
]

# Categorize instruments
instrument_columns = [
    'Bass Based', 'Guitar Based', 'Piano/Keyboard Based', 'Orchestral Strings',
    'Horns/Winds', 'Saxophone', 'Trumpet', 'Violin', 'Accordion', 'Banjo',
    'Bongos', 'Clarinet', 'Cowbell', 'Flute/Piccolo', 'Handclaps/Snaps',
    'Harmonica', 'Human Whistling', 'Kazoo', 'Mandolin', 'Pedal/Lap Steel',
    'Ocarina', 'Sitar', 'Ukulele'
]

# Calculate summary statistics for dashboard
vocal_summary = df[vocal_techniques].mean().sort_values(ascending=False)
instrument_prevalence = df[instrument_columns].mean().sort_values(ascending=False)
top_instruments = instrument_prevalence.head(10).index.tolist()

# Calculate correlations for insights
instrument_success_corr = df[top_instruments + ['Overall Rating', 'Weeks at Number One']].corr()[
    ['Overall Rating', 'Weeks at Number One']
].drop(['Overall Rating', 'Weeks at Number One'])

genre_vocal_analysis = df.groupby('CDR Genre')[vocal_techniques].mean()
genre_instrument_analysis = df.groupby('CDR Genre')[top_instruments].mean()

# Key metrics for the dashboard
def get_dashboard_metrics():
    total_songs = len(df)
    unique_artists = df['Artist'].nunique()
    avg_weeks_at_no1 = df['Weeks at Number One'].mean()
    avg_overall_rating = df['Overall Rating'].mean()
    
    # Vocal techniques metrics
    songs_with_falsetto = (df['Falsetto Vocal'] == 1).sum()
    songs_with_rap_verse = (df['Rap Verse in a Non-Rap Song'] == 1).sum()
    songs_with_spoken_word = (df['Spoken Word'] == 1).sum()
    explicit_songs = (df['Explicit'] == 1).sum()
    vocally_based_songs = (df['Vocally Based'] == 1).sum()
    
    # Instrumentation metrics
    guitar_based_songs = (df['Guitar Based'] == 1).sum()
    piano_based_songs = (df['Piano/Keyboard Based'] == 1).sum()
    bass_based_songs = (df['Bass Based'] == 1).sum()
    orchestral_songs = (df['Orchestral Strings'] == 1).sum()
    horn_wind_songs = (df['Horns/Winds'] == 1).sum()
    
    # Audio features metrics
    avg_energy = df['Energy'].mean()
    avg_danceability = df['Danceability'].mean()
    avg_bpm = df['BPM'].mean()
    avg_loudness = df['Loudness (dB)'].mean()
    
    # Calculate percentages
    pct_falsetto = (songs_with_falsetto / total_songs) * 100
    pct_rap_verse = (songs_with_rap_verse / total_songs) * 100
    pct_guitar = (guitar_based_songs / total_songs) * 100
    pct_piano = (piano_based_songs / total_songs) * 100
    pct_vocally_based = (vocally_based_songs / total_songs) * 100
    pct_explicit = (explicit_songs / total_songs) * 100
    
    # Genre analysis
    top_genres = df['CDR Genre'].value_counts().head(5)
    
    # Era analysis
    current_year = pd.to_datetime('today').year
    df['Era'] = pd.cut(df['Year'], 
                      bins=[1950, 1970, 1990, 2010, current_year],
                      labels=['50s-60s', '70s-80s', '90s-00s', '10s-Present'])
    
    era_vocal_trends = df.groupby('Era')[['Falsetto Vocal', 'Rap Verse in a Non-Rap Song', 'Explicit']].mean()
    
    return {
        'total_songs': total_songs,
        'unique_artists': unique_artists,
        'avg_weeks_at_no1': avg_weeks_at_no1,
        'avg_overall_rating': avg_overall_rating,
        'songs_with_falsetto': songs_with_falsetto,
        'songs_with_rap_verse': songs_with_rap_verse,
        'songs_with_spoken_word': songs_with_spoken_word,
        'explicit_songs': explicit_songs,
        'vocally_based_songs': vocally_based_songs,
        'guitar_based_songs': guitar_based_songs,
        'piano_based_songs': piano_based_songs,
        'bass_based_songs': bass_based_songs,
        'orchestral_songs': orchestral_songs,
        'horn_wind_songs': horn_wind_songs,
        'avg_energy': avg_energy,
        'avg_danceability': avg_danceability,
        'avg_bpm': avg_bpm,
        'avg_loudness': avg_loudness,
        'pct_falsetto': pct_falsetto,
        'pct_rap_verse': pct_rap_verse,
        'pct_guitar': pct_guitar,
        'pct_piano': pct_piano,
        'pct_vocally_based': pct_vocally_based,
        'pct_explicit': pct_explicit,
        'top_genres': top_genres,
        'era_vocal_trends': era_vocal_trends,
        'vocal_summary': vocal_summary,
        'instrument_prevalence': instrument_prevalence,
        'top_instruments': top_instruments
    }

# Get all metrics
metrics = get_dashboard_metrics()

# Print data quality check
print("Data Quality Check:")
print(f"Null values percentage: {(df.isnull().sum()/len(df)).mean():.2%}")
print(f"Dataset shape: {df.shape}")

# Object columns for reference
sbj_col = list(df.select_dtypes(include='object').columns)
print(f"\nObject columns: {len(sbj_col)}")
print(f"Numeric columns: {len(df.select_dtypes(include=['int64', 'float64']).columns)}")

# Print key insights
print("\n=== KEY INSIGHTS ===")
print(f"Most common vocal technique: {vocal_summary.index[0]} ({vocal_summary.iloc[0]:.1%})")
print(f"Most common instrument: {instrument_prevalence.index[0]} ({instrument_prevalence.iloc[0]:.1%})")
print(f"Top genre: {metrics['top_genres'].index[0]} ({metrics['top_genres'].iloc[0]} songs)")
print(f"Explicit content trend: {metrics['pct_explicit']:.1f}% of hit songs")