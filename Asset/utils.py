import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import textstat
from textblob import TextBlob
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

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

# Load and prepare data
df = pd.read_csv("c:\\Users\\Moving_King\\Documents\\BillBoard\\Billboard Hot 100 Number Ones Database - Data.csv")
#df = pd.read_csv('https://raw.githubusercontent.com/plotly/Figure-Friday/refs/heads/main/2025/week-44/Billboard%20Hot%20100%20Number%20Ones%20Database%20-%20Data.csv')
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

df['Year'] = pd.to_datetime(df['Date'], format='%d-%b-%y').dt.year

# Get unique years and sort them
unique_years = sorted(df['Year'].unique())

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

# ML Feature Engineering
class MusicMLAnalyzer:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.is_trained = False
        
    def prepare_ml_features(self):
        """Prepare features for machine learning"""
        # Select relevant features for ML
        ml_features = (vocal_techniques + top_instruments + 
                      ['Energy', 'Danceability', 'BPM', 'Loudness (dB)',
                       'Acousticness', 'Happiness', 'Length (Sec)'])
        
        # Create target variable (hit song classification)
        df['Hit_Score'] = (
            df['Overall Rating'] * 0.4 +
            df['Weeks at Number One'] * 0.3 +
            (5 - (df['Overall Rating'] - df['Overall Rating'].min()) / (df['Overall Rating'].max() - df['Overall Rating'].min()) * 5) * 0.3
        )
        df['Is_Hit'] = (df['Hit_Score'] > df['Hit_Score'].quantile(0.7)).astype(int)
        
        X = df[ml_features]
        y = df['Is_Hit']
        
        return X, y, ml_features
    
    def train_models(self):
        """Train multiple ML models"""
        try:
            X, y, features = self.prepare_ml_features()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Random Forest Classifier
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            self.models['random_forest'] = rf_model
            
            # Feature importance
            self.feature_importance['random_forest'] = pd.DataFrame({
                'feature': features,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Calculate accuracy
            y_pred = rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            return accuracy
        except Exception as e:
            print(f"Error training models: {e}")
            return 0.0
    
    def predict_hit_probability(self, features_dict):
        """Predict hit probability for new song features"""
        if not self.is_trained:
            self.train_models()
        
        try:
            # Convert input to dataframe
            input_features = pd.DataFrame([features_dict])
            
            # Get feature list from training
            X, _, _ = self.prepare_ml_features()
            
            # Ensure all required features are present
            for col in X.columns:
                if col not in input_features.columns:
                    # Use median values for missing features
                    input_features[col] = X[col].median()
            
            # Reorder columns to match training data
            input_features = input_features[X.columns]
            
            # Predict probability
            probability = self.models['random_forest'].predict_proba(input_features)[0][1]
            return probability
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 0.5

# NLP Analysis
class LyricalAnalyzer:
    def __init__(self):
        self.sentiment_results = None
        self.complexity_metrics = None
        
    def analyze_sentiment(self, lyrics):
        """Analyze sentiment of lyrics"""
        if pd.isna(lyrics) or lyrics == '':
            return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutral'}
        
        try:
            blob = TextBlob(str(lyrics))
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
                
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment
            }
        except:
            return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutral'}
    
    def calculate_complexity(self, lyrics):
        """Calculate lyrical complexity metrics"""
        if pd.isna(lyrics) or lyrics == '':
            return {
                'word_count': 0,
                'unique_words': 0,
                'vocab_richness': 0,
                'avg_word_length': 0,
                'readability_score': 0
            }
        
        try:
            text = str(lyrics)
            words = re.findall(r'\b\w+\b', text.lower())
            word_count = len(words)
            unique_words = len(set(words))
            vocab_richness = unique_words / word_count if word_count > 0 else 0
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            # Simple readability score
            readability = textstat.flesch_reading_ease(text) if text.strip() else 0
            
            return {
                'word_count': word_count,
                'unique_words': unique_words,
                'vocab_richness': vocab_richness,
                'avg_word_length': avg_word_length,
                'readability_score': readability
            }
        except:
            return {
                'word_count': 0,
                'unique_words': 0,
                'vocab_richness': 0,
                'avg_word_length': 0,
                'readability_score': 0
            }
    
    def analyze_all_lyrics(self):
        """Analyze all lyrics in the dataset"""
        sentiment_data = []
        complexity_data = []
        
        for idx, row in df.iterrows():
            lyrics = row.get('Lyrics', '')
            
            # Sentiment analysis
            sentiment = self.analyze_sentiment(lyrics)
            sentiment_data.append({
                'Song': row['Song'],
                'Artist': row['Artist'],
                'Year': row['Year'],
                'polarity': sentiment['polarity'],
                'subjectivity': sentiment['subjectivity'],
                'sentiment': sentiment['sentiment']
            })
            
            # Complexity analysis
            complexity = self.calculate_complexity(lyrics)
            complexity_data.append({
                'Song': row['Song'],
                'Artist': row['Artist'],
                'Year': row['Year'],
                **complexity
            })
        
        self.sentiment_results = pd.DataFrame(sentiment_data)
        self.complexity_metrics = pd.DataFrame(complexity_data)
        
        return self.sentiment_results, self.complexity_metrics

# Recommendation System
class MusicRecommender:
    def __init__(self):
        self.similarity_matrix = None
        self.feature_columns = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_recommendation_features(self):
        """Prepare features for recommendation system"""
        # Combine musical features for similarity calculation
        feature_columns = (vocal_techniques + top_instruments + 
                          ['Energy', 'Danceability', 'BPM', 'Loudness (dB)',
                           'Acousticness', 'Happiness'])
        
        # Add genre encoding
        genre_encoder = LabelEncoder()
        df['Genre_encoded'] = genre_encoder.fit_transform(df['CDR Genre'])
        feature_columns.append('Genre_encoded')
        
        # Add era feature
        current_year = pd.to_datetime('today').year
        df['Era'] = pd.cut(df['Year'], 
                          bins=[1950, 1970, 1990, 2010, current_year],
                          labels=['50s-60s', '70s-80s', '90s-00s', '10s-Present'])
        era_encoder = LabelEncoder()
        df['Era_encoded'] = era_encoder.fit_transform(df['Era'])
        feature_columns.append('Era_encoded')
        
        self.feature_columns = feature_columns
        return df[feature_columns]
    
    def build_similarity_matrix(self):
        """Build cosine similarity matrix for all songs"""
        try:
            features = self.prepare_recommendation_features()
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Calculate cosine similarity
            self.similarity_matrix = cosine_similarity(scaled_features)
            self.is_trained = True
            
            print("Similarity matrix built successfully!")
            return self.similarity_matrix
            
        except Exception as e:
            print(f"Error building similarity matrix: {e}")
            return None
    
    def get_song_recommendations(self, song_name, artist_name=None, n_recommendations=10):
        """Get song recommendations based on musical similarity"""
        if not self.is_trained:
            self.build_similarity_matrix()
        
        try:
            # Find the target song
            if artist_name:
                target_mask = (df['Song'] == song_name) & (df['Artist'] == artist_name)
            else:
                target_mask = (df['Song'] == song_name)
            
            if not target_mask.any():
                # Try fuzzy matching
                similar_songs = df[df['Song'].str.contains(song_name, case=False, na=False)]
                if len(similar_songs) > 0:
                    target_idx = similar_songs.index[0]
                else:
                    return None
            else:
                target_idx = df[target_mask].index[0]
            
            # Get similarity scores for target song
            similarity_scores = list(enumerate(self.similarity_matrix[target_idx]))
            
            # Sort by similarity score (descending)
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            
            # Get top N recommendations (excluding the song itself)
            recommendations = []
            for idx, score in similarity_scores[1:n_recommendations+1]:
                song_data = df.iloc[idx]
                recommendations.append({
                    'Song': song_data['Song'],
                    'Artist': song_data['Artist'],
                    'Year': song_data['Year'],
                    'CDR Genre': song_data['CDR Genre'],
                    'Similarity Score': round(score, 3),
                    'Overall Rating': song_data['Overall Rating'],
                    'Energy': song_data['Energy'],
                    'Danceability': song_data['Danceability'],
                    'BPM': song_data['BPM']
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return None
    
    def get_recommendations_by_features(self, feature_preferences, n_recommendations=10):
        """Get recommendations based on feature preferences"""
        if not self.is_trained:
            self.build_similarity_matrix()
        
        try:
            # Create a virtual song based on preferences
            virtual_song = pd.DataFrame([{col: 0 for col in self.feature_columns}])
            
            # Set preferred features
            for feature, value in feature_preferences.items():
                if feature in virtual_song.columns:
                    virtual_song[feature] = value
            
            # Fill missing features with dataset averages
            for col in self.feature_columns:
                if col not in feature_preferences:
                    virtual_song[col] = df[col].mean()
            
            # Scale the virtual song
            scaled_virtual = self.scaler.transform(virtual_song)
            
            # Calculate similarity with all songs
            features_scaled = self.scaler.transform(df[self.feature_columns])
            similarities = cosine_similarity(scaled_virtual, features_scaled)[0]
            
            # Get top recommendations
            similar_indices = similarities.argsort()[::-1][:n_recommendations]
            
            recommendations = []
            for idx in similar_indices:
                song_data = df.iloc[idx]
                recommendations.append({
                    'Song': song_data['Song'],
                    'Artist': song_data['Artist'],
                    'Year': song_data['Year'],
                    'CDR Genre': song_data['CDR Genre'],
                    'Similarity Score': round(similarities[idx], 3),
                    'Overall Rating': song_data['Overall Rating'],
                    'Energy': song_data['Energy'],
                    'Danceability': song_data['Danceability'],
                    'BPM': song_data['BPM']
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Error in feature-based recommendations: {e}")
            return None
    
    def get_cluster_recommendations(self, n_clusters=5):
        """Cluster songs and analyze cluster characteristics"""
        try:
            features = self.prepare_recommendation_features()
            scaled_features = self.scaler.fit_transform(features)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['Cluster'] = kmeans.fit_predict(scaled_features)
            
            # Analyze cluster characteristics
            cluster_analysis = df.groupby('Cluster').agg({
                'Overall Rating': 'mean',
                'Energy': 'mean',
                'Danceability': 'mean',
                'BPM': 'mean',
                'CDR Genre': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
                'Song': 'count'
            }).rename(columns={'Song': 'Count'})
            
            return cluster_analysis, df[['Song', 'Artist', 'Cluster', 'CDR Genre', 'Overall Rating']]
            
        except Exception as e:
            print(f"Error in clustering: {e}")
            return None, None

# Audio Feature Analysis
class AudioFeatureEngineer:
    def __init__(self):
        self.audio_clusters = {}
        
    def create_audio_profile(self):
        """Create comprehensive audio feature profiles"""
        audio_features = ['Energy', 'Danceability', 'BPM', 'Loudness (dB)', 
                         'Acousticness', 'Happiness']
        
        # Calculate audio feature correlations
        audio_corr = df[audio_features + ['Overall Rating']].corr()['Overall Rating'].drop('Overall Rating')
        
        # Create audio complexity score
        df['Audio_Complexity'] = (
            df['Energy'] + df['Danceability'] + 
            (df['BPM'] / 200) +  # Normalize BPM
            (abs(df['Loudness (dB)']) / 60)  # Normalize loudness
        )
        
        # Genre audio signatures
        genre_audio_profiles = df.groupby('CDR Genre')[audio_features].mean()
        
        return {
            'audio_correlations': audio_corr,
            'audio_complexity': df['Audio_Complexity'],
            'genre_profiles': genre_audio_profiles
        }

# Initialize analyzers
ml_analyzer = MusicMLAnalyzer()
lyrical_analyzer = LyricalAnalyzer()
recommender = MusicRecommender()
audio_engineer = AudioFeatureEngineer()

# Train models and perform analyses
print("Training ML models...")
ml_accuracy = ml_analyzer.train_models()
print(f"ML Model Accuracy: {ml_accuracy:.2%}")

print("Analyzing lyrics...")
sentiment_results, complexity_metrics = lyrical_analyzer.analyze_all_lyrics()

print("Building recommendation system...")
similarity_matrix = recommender.build_similarity_matrix()

print("Engineering audio features...")
audio_analysis = audio_engineer.create_audio_profile()

# Merge analysis results with main dataframe
df = df.merge(sentiment_results[['Song', 'Artist', 'polarity', 'subjectivity', 'sentiment']], 
              on=['Song', 'Artist'], how='left')
df = df.merge(complexity_metrics[['Song', 'Artist', 'word_count', 'vocab_richness', 'readability_score']], 
              on=['Song', 'Artist'], how='left')

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
    avg_energy = (df['Energy'].mean() / df['Energy'].max()) * 100
    avg_danceability = (df['Danceability'].mean()/ df['Danceability'].max()) * 100
    avg_bpm = df['BPM'].mean()
    avg_loudness = df['Loudness (dB)'].mean()
    
    # NLP metrics
    avg_sentiment = df['polarity'].mean()
    avg_complexity = df['vocab_richness'].mean() * 100
    
    # Calculate percentages
    pct_falsetto = (songs_with_falsetto / total_songs) * 100
    pct_rap_verse = (songs_with_rap_verse / total_songs) * 100
    pct_guitar = (guitar_based_songs / total_songs) * 100
    pct_piano = (piano_based_songs / total_songs) * 100
    pct_vocally_based = (vocally_based_songs / total_songs) * 100
    pct_explicit = (explicit_songs / total_songs) * 100
    
    # Genre analysis
    top_genres = df['CDR Genre'].value_counts().head(5)
    
    # ML metrics
    hit_songs = df['Is_Hit'].sum()
    pct_hits = (hit_songs / total_songs) * 100
    
    # Recommendation system metrics
    cluster_analysis, _ = recommender.get_cluster_recommendations(n_clusters=5)
    
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
        'avg_sentiment': avg_sentiment,
        'avg_complexity': avg_complexity,
        'pct_falsetto': pct_falsetto,
        'pct_rap_verse': pct_rap_verse,
        'pct_guitar': pct_guitar,
        'pct_piano': pct_piano,
        'pct_vocally_based': pct_vocally_based,
        'pct_explicit': pct_explicit,
        'pct_hits': pct_hits,
        'hit_songs': hit_songs,
        'top_genres': top_genres,
        'ml_accuracy': ml_accuracy,
        'feature_importance': ml_analyzer.feature_importance['random_forest'],
        'sentiment_trends': lyrical_analyzer.sentiment_results,
        'audio_analysis': audio_analysis,
        'cluster_analysis': cluster_analysis,
        'recommender': recommender
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
print(f"ML Model Accuracy: {metrics['ml_accuracy']:.2%}")
print(f"Average lyrical sentiment: {metrics['avg_sentiment']:.2f}")
print(f"Recommendation system ready: {recommender.is_trained}")