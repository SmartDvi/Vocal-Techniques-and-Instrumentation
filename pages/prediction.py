import pandas as pd
import dash_mantine_components as dmc
from dash import Input, Output, dcc, callback, State, html
import dash
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Asset.utils import colors, df, vocal_techniques, top_instruments, metrics
from Asset.utils import ml_analyzer, lyrical_analyzer
from dash_iconify import DashIconify

dash.register_page(
    __name__,
    path='/Prediction',
    title=' Music Prediction Insights', 
    description='Machine Learning and NLP analysis of hit song patterns',
    order=3
)

# Helper function to apply filters (same as in Data_Insight.py)
def apply_filters(dataframe, filter_data, year_range):
    """Apply all filters from filter store and year range"""
    if not year_range or len(year_range) != 2:
        year_range = [int(df['Year'].min()), int(df['Year'].max())]
    
    filtered_df = dataframe[
        (dataframe['Year'] >= year_range[0]) & 
        (dataframe['Year'] <= year_range[1])
    ]
    
    # Apply filters from filter store
    if filter_data:
        if filter_data.get('energy_levels'):
            filtered_df = filtered_df[filtered_df['Energy_level'].isin(filter_data['energy_levels'])]
        
        if filter_data.get('danceability_levels'):
            filtered_df = filtered_df[filtered_df['Danceability_level'].isin(filter_data['danceability_levels'])]
        
        if filter_data.get('genres'):
            filtered_df = filtered_df[filtered_df['CDR Genre'].isin(filter_data['genres'])]
        
        if filter_data.get('producers'):
            filtered_df = filtered_df[filtered_df['Producers'].isin(filter_data['producers'])]
    
    return filtered_df

# Helper function for empty figures
def create_empty_figure(message="No data available"):
    """Create an empty figure with a message"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(color='white', size=14)
    )
    fig.update_layout(
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white',
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

# Year range slider for this page
year_range_slider = dcc.RangeSlider(
    id='year-range-slider-prediction',
    min=int(df['Year'].min()),
    max=int(df['Year'].max()),
    value=[int(df['Year'].min()), int(df['Year'].max())],
    marks={str(year): str(year) for year in range(int(df['Year'].min()), int(df['Year'].max())+1, 10)},
    step=1,
    tooltip={"placement": "bottom", "always_visible": True}
)

# Create main tabs for prediction page
prediction_tabs = dmc.Tabs(
    [
        dmc.TabsList(
            [
                dmc.TabsTab("Hit Prediction", value="prediction", color='red'),
                dmc.TabsTab("Feature Analysis", value="features", color='blue'),
                dmc.TabsTab("NLP Insights", value="nlp", color='green'),
                dmc.TabsTab("Audio Intelligence", value="audio", color='purple'),
            ]
        ),
        dmc.TabsPanel(
            # Hit Prediction Tab
            dmc.Grid([
                dmc.GridCol(span=6, children=[
                    dmc.Paper([
                        dmc.Title("🎵 Hit Song Predictor", order=3, mb="md", c='white'),
                        dmc.Text("Enter song features to predict hit probability", size="sm", c="dimmed", mb="lg"),
                        
                        dmc.Stack([
                            dmc.NumberInput(
                                label="Energy Level (0-100)",
                                id="energy-input",
                                value=70,
                                min=0,
                                max=100,
                                step=1,
                                description="How energetic is the song?"
                            ),
                            dmc.NumberInput(
                                label="Danceability (0-100)",
                                id="danceability-input",
                                value=65,
                                min=0,
                                max=100,
                                step=1,
                                description="How danceable is the song?"
                            ),
                            dmc.NumberInput(
                                label="BPM (Beats Per Minute)",
                                id="bpm-input",
                                value=120,
                                min=60,
                                max=200,
                                step=5,
                                description="Song tempo"
                            ),
                            dmc.NumberInput(
                                label="Loudness (dB)",
                                id="loudness-input",
                                value=-8,
                                min=-60,
                                max=0,
                                step=1,
                                description="Volume intensity"
                            ),
                            dmc.Select(
                                label="Primary Instrument",
                                id="instrument-input",
                                data=[{'label': inst, 'value': inst} for inst in top_instruments[:8]],
                                value='Guitar Based',
                                description="Main instrument in the song"
                            ),
                            dmc.Select(
                                label="Vocal Technique",
                                id="vocal-input",
                                data=[{'label': tech, 'value': tech} for tech in vocal_techniques[:6]],
                                value='Vocally Based',
                                description="Primary vocal style"
                            ),
                            dmc.Button(
                                "Predict Hit Probability", 
                                id="predict-btn", 
                                color="red", 
                                size="lg",
                                leftSection=dmc.ThemeIcon(
                                    DashIconify(icon="radix-icons:lightning-bolt", width=20), variant="filled")
                            ),
                        ], gap="md")
                    ], p="lg", style={'backgroundColor': colors['card_bg']}, withBorder=True)
                ]),
                dmc.GridCol(span=6, children=[
                    dmc.Paper([
                        dmc.Title("Prediction Results", order=3, mb="md", c='white'),
                        html.Div(id="prediction-output", style={
                            'textAlign': 'center', 
                            'padding': '40px',
                            'minHeight': '300px',
                            'display': 'flex',
                            'flexDirection': 'column',
                            'justifyContent': 'center',
                            'alignItems': 'center'
                        }),
                        dmc.Text("Based on analysis of Billboard Hot 100 patterns", size="sm", c="dimmed", ta="center")
                    ], p="lg", style={'backgroundColor': colors['card_bg']}, withBorder=True)
                ]),
                dmc.GridCol(span=12, children=[
                    dmc.Paper([
                        dmc.Title("Model Performance", order=4, mb="md", c='white'),
                        dmc.Grid([
                            dmc.GridCol(span=3, children=dmc.Card(
                                children=[
                                    dmc.Text("Accuracy", size="sm", c="dimmed"),
                                    dmc.Text(f"{metrics.get('ml_accuracy', 0):.1%}", size="xl", fw=700, c=colors['success']),
                                    dmc.Progress(value=metrics.get('ml_accuracy', 0)*100, color="green", size="md")
                                ],
                                withBorder=True,
                                p="md"
                            )),
                            dmc.GridCol(span=3, children=dmc.Card(
                                children=[
                                    dmc.Text("Hit Songs", size="sm", c="dimmed"),
                                    dmc.Text(f"{metrics.get('hit_songs', 0)}", size="xl", fw=700, c=colors['primary']),
                                    dmc.Text(f"{metrics.get('pct_hits', 0):.1f}% of dataset", size="sm")
                                ],
                                withBorder=True,
                                p="md"
                            )),
                            dmc.GridCol(span=3, children=dmc.Card(
                                children=[
                                    dmc.Text("Features Used", size="sm", c="dimmed"),
                                    dmc.Text(f"{len(metrics.get('feature_importance', pd.DataFrame()))}", size="xl", fw=700, c=colors['warning']),
                                    dmc.Text("Musical elements", size="sm")
                                ],
                                withBorder=True,
                                p="md"
                            )),
                            dmc.GridCol(span=3, children=dmc.Card(
                                children=[
                                    dmc.Text("Training Data", size="sm", c="dimmed"),
                                    dmc.Text(f"{len(df)}", size="xl", fw=700, c=colors['secondary']),
                                    dmc.Text("Billboard hits", size="sm")
                                ],
                                withBorder=True,
                                p="md"
                            )),
                        ])
                    ], p="lg", style={'backgroundColor': colors['card_bg']})
                ])
            ]),
            value="prediction"
        ),
        dmc.TabsPanel(
            # Feature Analysis Tab
            dmc.Grid([
                dmc.GridCol(span=6, children=[
                    dmc.Paper([
                        dmc.Title("Feature Importance", order=4, mb="md", c='white'),
                        dcc.Graph(id="feature-importance-chart"),
                        dmc.Text("Top factors influencing hit song success", size="sm", c="dimmed")
                    ], p="md", style={'backgroundColor': colors['card_bg']})
                ]),
                dmc.GridCol(span=6, children=[
                    dmc.Paper([
                        dmc.Title("Feature Correlations", order=4, mb="md", c='white'),
                        dcc.Graph(id="feature-correlation-chart"),
                        dmc.Text("Relationships between musical elements", size="sm", c="dimmed")
                    ], p="md", style={'backgroundColor': colors['card_bg']})
                ]),
                dmc.GridCol(span=12, children=[
                    dmc.Paper([
                        dmc.Title("Feature Trends Over Time", order=4, mb="md", c='white'),
                        dcc.Graph(id="feature-trends-chart"),
                        dmc.Text("Evolution of important features across decades", size="sm", c="dimmed")
                    ], p="md", style={'backgroundColor': colors['card_bg']})
                ])
            ]),
            value="features"
        ),
        dmc.TabsPanel(
            # NLP Insights Tab
            dmc.Grid([
                dmc.GridCol(span=6, children=[
                    dmc.Paper([
                        dmc.Title("Lyrical Sentiment Trends", order=4, mb="md", c='white'),
                        dcc.Graph(id="sentiment-trends-chart"),
                        dmc.Text("Evolution of emotional content in lyrics", size="sm", c="dimmed")
                    ], p="md", style={'backgroundColor': colors['card_bg']})
                ]),
                dmc.GridCol(span=6, children=[
                    dmc.Paper([
                        dmc.Title("Lyrical Complexity", order=4, mb="md", c='white'),
                        dcc.Graph(id="complexity-trends-chart"),
                        dmc.Text("Vocabulary richness over time", size="sm", c="dimmed")
                    ], p="md", style={'backgroundColor': colors['card_bg']})
                ]),
                dmc.GridCol(span=12, children=[
                    dmc.Paper([
                        dmc.Title("Sentiment vs Success", order=4, mb="md", c='white'),
                        dcc.Graph(id="sentiment-success-chart"),
                        dmc.Text("Relationship between lyrical emotion and chart performance", size="sm", c="dimmed")
                    ], p="md", style={'backgroundColor': colors['card_bg']})
                ])
            ]),
            value="nlp"
        ),
        dmc.TabsPanel(
            # Audio Intelligence Tab
            dmc.Grid([
                dmc.GridCol(span=6, children=[
                    dmc.Paper([
                        dmc.Title("Audio Feature Clusters", order=4, mb="md", c='white'),
                        dcc.Graph(id="audio-cluster-chart"),
                        dmc.Text("Natural groupings based on audio characteristics", size="sm", c="dimmed")
                    ], p="md", style={'backgroundColor': colors['card_bg']})
                ]),
                dmc.GridCol(span=6, children=[
                    dmc.Paper([
                        dmc.Title("Genre Audio Profiles", order=4, mb="md", c='white'),
                        dcc.Graph(id="genre-audio-chart"),
                        dmc.Text("Distinctive audio features by genre", size="sm", c="dimmed")
                    ], p="md", style={'backgroundColor': colors['card_bg']})
                ]),
                dmc.GridCol(span=12, children=[
                    dmc.Paper([
                        dmc.Title("Audio Evolution", order=4, mb="md", c='white'),
                        dcc.Graph(id="audio-evolution-chart"),
                        dmc.Text("How audio features have changed over decades", size="sm", c="dimmed")
                    ], p="md", style={'backgroundColor': colors['card_bg']})
                ])
            ]),
            value="audio"
        ),
    ],
    value="prediction",
    id="prediction-tabs"
)

layout = dmc.MantineProvider(
    children=[
        dmc.Container(
            fluid=True,
            px='xl',
            children=[
                dmc.Text('AI Music Intelligence', tt="uppercase", size="xl", c="blue", 
                        ta="center", td="underline", fw=700, p=20),
                
                # Filters Section
                dmc.Paper(
                    p="md",
                    m='md',
                    shadow='sm',
                    children=[
                        dmc.Grid([
                            dmc.GridCol(span=12, children=dmc.Text("Adjust Year Range for Analysis:", size="sm", fw=500)),
                        ]),
                        dmc.Grid([
                            dmc.GridCol(span=12, children=year_range_slider)
                        ]),
                        dmc.Grid([
                            dmc.GridCol(span=12, children=dmc.Text(
                                "Global filters from sidebar apply to all analysis below", 
                                size="sm", c="dimmed", ta="center"
                            ))
                        ])
                    ],
                    style={'backgroundColor': colors['card_bg']}
                ),
                
                # Main Prediction Content
                dmc.Paper(
                    p=0,
                    m='md',
                    shadow='md',
                    children=[
                        prediction_tabs
                    ]
                )
            ]
        )
    ]
)

# PREDICTION CALLBACKS
@callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("energy-input", "value"),
    State("danceability-input", "value"),
    State("bpm-input", "value"),
    State("loudness-input", "value"),
    State("instrument-input", "value"),
    State("vocal-input", "value"),
    prevent_initial_call=True
)
def predict_hit_probability(n_clicks, energy, danceability, bpm, loudness, instrument, vocal):
    """Predict hit probability based on input features"""
    if n_clicks is None:
        return dmc.Stack([
            dmc.Text("🎵", size="48px", ta="center"),
            dmc.Text("Enter song features and click predict to see results", size="lg", c="dimmed", ta="center"),
            dmc.Text("The model analyzes patterns from Billboard Hot 100 hits", size="sm", c="dimmed", ta="center")
        ], align="center", gap="xs")
    
    try:
        # Prepare features for prediction (convert to model scale)
        features = {
            'Energy': energy / 100.0,  # Convert to 0-1 scale
            'Danceability': danceability / 100.0,  # Convert to 0-1 scale
            'BPM': bpm,
            'Loudness (dB)': loudness,
            instrument: 1,  # Set the selected instrument to 1
            vocal: 1,       # Set the selected vocal technique to 1
        }
        
        # Set other instruments and vocals to 0
        for inst in top_instruments:
            if inst != instrument:
                features[inst] = 0
        for tech in vocal_techniques:
            if tech != vocal:
                features[tech] = 0
        
        # Add default values for other required features
        features['Acousticness'] = df['Acousticness'].median() if 'Acousticness' in df.columns else 0.5
        features['Happiness'] = df['Happiness'].median() if 'Happiness' in df.columns else 0.5
        features['Length (Sec)'] = df['Length (Sec)'].median() if 'Length (Sec)' in df.columns else 180
        
        # Get prediction
        probability = ml_analyzer.predict_hit_probability(features)
        
        # Create visual output based on probability
        if probability >= 0.7:
            color = "green"
            emoji = "🎉"
            message = "HIGH HIT POTENTIAL"
            advice = "This combination has strong hit characteristics!"
            confidence = "Very High"
        elif probability >= 0.6:
            color = "teal"
            emoji = "✨"
            message = "STRONG POTENTIAL"
            advice = "Excellent foundation for a hit song"
            confidence = "High"
        elif probability >= 0.5:
            color = "yellow"
            emoji = "👍"
            message = "MODERATE POTENTIAL"
            advice = "Good foundation, consider optimizing features"
            confidence = "Moderate"
        elif probability >= 0.4:
            color = "orange"
            emoji = "💡"
            message = "DEVELOPMENT NEEDED"
            advice = "Consider adjusting musical elements"
            confidence = "Low"
        else:
            color = "red"
            emoji = "🎵"
            message = "EXPERIMENTAL COMBINATION"
            advice = "Try different feature combinations"
            confidence = "Very Low"
        
        return dmc.Stack([
            dmc.Text(emoji, size="48px", ta="center"),
            dmc.Text(f"{probability:.1%}", size="42px", fw=900, c=color, ta="center"),
            dmc.Text(message, size="lg", fw=700, c=color, ta="center"),
            dmc.Text(f"Confidence: {confidence}", size="sm", c=color, ta="center"),
            dmc.Progress(
                value=probability*100, 
                color=color, 
                size="xl", 
                radius="xl", 
                style={'width': '80%', 'marginTop': '10px'}
            ),
            dmc.Text(advice, size="sm", c="dimmed", ta="center", style={'marginTop': '15px'}),
            dmc.Text(f"Based on {len(df)} Billboard hits", size="xs", c="dimmed", ta="center")
        ], align="center", gap="xs")
        
    except Exception as e:
        return dmc.Stack([
            dmc.Text("❌", size="48px", ta="center"),
            dmc.Text("Prediction Error", size="lg", c="red", ta="center"),
            dmc.Text("Please try different values", size="sm", c="dimmed", ta="center"),
            dmc.Text(str(e), size="xs", c="red", ta="center")
        ], align="center", gap="xs")

# FEATURE ANALYSIS CALLBACKS
@callback(
    Output("feature-importance-chart", "figure"),
    Input('filter-store', 'data'),
    Input('year-range-slider-prediction', 'value')
)
def update_feature_importance(filter_data, year_range):
    """Update feature importance chart based on filters"""
    filtered_df = apply_filters(df, filter_data, year_range)
    
    # Use pre-trained feature importance
    feature_importance = metrics['feature_importance']
    
    if feature_importance.empty:
        # Create fallback feature importance
        feature_importance = pd.DataFrame({
            'feature': ['Energy', 'Danceability', 'BPM', 'Overall Rating', 'Guitar Based'],
            'importance': [0.25, 0.20, 0.15, 0.20, 0.20]
        })
    
    # Take top 10 features
    top_features = feature_importance.head(10)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title='Top Features Influencing Hit Song Success',
        labels={'importance': 'Feature Importance', 'feature': ''},
        color='importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white',
        showlegend=False,
        height=400,
        xaxis=dict(range=[0, top_features['importance'].max() * 1.1])
    )
    
    return fig

@callback(
    Output("feature-correlation-chart", "figure"),
    Input('filter-store', 'data'),
    Input('year-range-slider-prediction', 'value')
)
def update_feature_correlations(filter_data, year_range):
    """Update feature correlation heatmap"""
    filtered_df = apply_filters(df, filter_data, year_range)
    
    # Select top features for correlation
    top_features = metrics['feature_importance'].head(6)['feature'].tolist() if not metrics['feature_importance'].empty else ['Energy', 'Danceability', 'BPM']
    
    # Ensure we have valid features
    available_features = [f for f in top_features if f in filtered_df.columns]
    if 'Overall Rating' not in available_features and 'Overall Rating' in filtered_df.columns:
        available_features.append('Overall Rating')
    
    if len(available_features) < 2:
        # Fallback features
        available_features = ['Energy', 'Danceability', 'BPM', 'Overall Rating']
        available_features = [f for f in available_features if f in filtered_df.columns]
    
    try:
        correlation_matrix = filtered_df[available_features].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title='Feature Correlations with Overall Rating',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            zmin=-1,
            zmax=1
        )
        
        fig.update_layout(
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font_color='white',
            height=400
        )
        
        return fig
    except Exception as e:
        # Return empty figure if correlation fails
        return create_empty_figure("Insufficient data for correlation analysis")

@callback(
    Output("feature-trends-chart", "figure"),
    Input('filter-store', 'data'),
    Input('year-range-slider-prediction', 'value')
)
def update_feature_trends(filter_data, year_range):
    """Update feature trends over time"""
    filtered_df = apply_filters(df, filter_data, year_range)
    
    if len(filtered_df) == 0:
        return create_empty_figure("No data available for selected filters")
    
    # Get top 3 features from feature importance
    if not metrics['feature_importance'].empty:
        top_features = metrics['feature_importance'].head(3)['feature'].tolist()
    else:
        # Fallback features if feature importance is not available
        top_features = ['Energy', 'Danceability', 'BPM']
    
    # Ensure features exist in dataframe
    available_features = [f for f in top_features if f in filtered_df.columns]
    
    if len(available_features) == 0:
        # Final fallback
        available_features = ['Energy', 'Danceability']
        available_features = [f for f in available_features if f in filtered_df.columns]
    
    # Group by year and calculate average feature values
    yearly_trends = filtered_df.groupby('Year')[available_features].mean().reset_index()
    
    fig = go.Figure()
    
    colors_list = ['#EF553B', '#00CC96', '#636EFA']  # Red, Green, Blue
    
    for i, feature in enumerate(available_features):
        fig.add_trace(go.Scatter(
            x=yearly_trends['Year'],
            y=yearly_trends[feature],
            name=feature,
            mode='lines+markers',
            line=dict(color=colors_list[i % len(colors_list)], width=3),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title='Top Feature Trends Over Time',
        xaxis_title='Year',
        yaxis_title='Feature Value',
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# NLP ANALYSIS CALLBACKS
@callback(
    Output("sentiment-trends-chart", "figure"),
    Input('filter-store', 'data'),
    Input('year-range-slider-prediction', 'value')
)
def update_sentiment_trends(filter_data, year_range):
    """Update sentiment trends over time"""
    filtered_df = apply_filters(df, filter_data, year_range)
    
    if len(filtered_df) == 0:
        return create_empty_figure("No data available for selected filters")
    
    # Group by year and calculate average sentiment
    yearly_sentiment = filtered_df.groupby('Year')['polarity'].mean().reset_index()
    
    fig = px.line(
        yearly_sentiment,
        x='Year',
        y='polarity',
        title='Lyrical Sentiment Trends Over Time',
        labels={'polarity': 'Average Sentiment Score', 'Year': 'Year'}
    )
    
    fig.update_layout(
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white',
        height=400
    )
    
    # Add reference line at 0 and styling
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
    fig.update_traces(line=dict(color='#FF6B6B', width=3), marker=dict(color='#FF6B6B', size=6))
    
    return fig

@callback(
    Output("complexity-trends-chart", "figure"),
    Input('filter-store', 'data'),
    Input('year-range-slider-prediction', 'value')
)
def update_complexity_trends(filter_data, year_range):
    """Update lyrical complexity trends"""
    filtered_df = apply_filters(df, filter_data, year_range)
    
    if len(filtered_df) == 0:
        return create_empty_figure("No data available for selected filters")
    
    # Group by year and calculate average complexity
    yearly_complexity = filtered_df.groupby('Year')['vocab_richness'].mean().reset_index()
    
    fig = px.line(
        yearly_complexity,
        x='Year',
        y='vocab_richness',
        title='Lyrical Complexity Trends Over Time',
        labels={'vocab_richness': 'Vocabulary Richness', 'Year': 'Year'}
    )
    
    fig.update_layout(
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white',
        height=400
    )
    
    # Add styling
    fig.update_traces(line=dict(color='#4ECDC4', width=3), marker=dict(color='#4ECDC4', size=6))
    
    return fig

@callback(
    Output("sentiment-success-chart", "figure"),
    Input('filter-store', 'data'),
    Input('year-range-slider-prediction', 'value')
)
def update_sentiment_success(filter_data, year_range):
    """Update sentiment vs success relationship"""
    filtered_df = apply_filters(df, filter_data, year_range)
    
    if len(filtered_df) == 0:
        return create_empty_figure("No data available for selected filters")
    
    # Sample data for better performance if dataset is large
    if len(filtered_df) > 1000:
        plot_df = filtered_df.sample(1000, random_state=42)
    else:
        plot_df = filtered_df
    
    fig = px.scatter(
        plot_df,
        x='polarity',
        y='Overall Rating',
        color='CDR Genre',
        title='Lyrical Sentiment vs Chart Success',
        labels={'polarity': 'Sentiment Score', 'Overall Rating': 'Overall Rating'},
        size='Weeks at Number One',
        hover_data=['Song', 'Artist'],
        opacity=0.7
    )
    
    fig.update_layout(
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white',
        height=500
    )
    
    return fig

# AUDIO ANALYSIS CALLBACKS
@callback(
    Output("audio-evolution-chart", "figure"),
    Input('filter-store', 'data'),
    Input('year-range-slider-prediction', 'value')
)
def update_audio_evolution(filter_data, year_range):
    """Update audio feature evolution over time"""
    filtered_df = apply_filters(df, filter_data, year_range)
    
    if len(filtered_df) == 0:
        return create_empty_figure("No data available for selected filters")
    
    # Group by year and calculate average audio features
    audio_features = ['Energy', 'Danceability', 'BPM']
    available_features = [f for f in audio_features if f in filtered_df.columns]
    
    if len(available_features) == 0:
        return create_empty_figure("No audio features available")
    
    yearly_audio = filtered_df.groupby('Year')[available_features].mean().reset_index()
    
    fig = go.Figure()
    
    colors_list = ['#EF553B', '#00CC96', '#636EFA', '#AB63FA', '#FFA15A']
    
    for i, feature in enumerate(available_features):
        fig.add_trace(go.Scatter(
            x=yearly_audio['Year'],
            y=yearly_audio[feature],
            name=feature,
            mode='lines+markers',
            line=dict(color=colors_list[i % len(colors_list)], width=3),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title='Audio Feature Evolution Over Time',
        xaxis_title='Year',
        yaxis_title='Feature Value',
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

@callback(
    Output("audio-cluster-chart", "figure"),
    Input('filter-store', 'data'),
    Input('year-range-slider-prediction', 'value')
)
def update_audio_clusters(filter_data, year_range):
    """Update audio feature clustering visualization"""
    filtered_df = apply_filters(df, filter_data, year_range)
    
    if len(filtered_df) == 0:
        return create_empty_figure("No data available for selected filters")
    
    # Use Energy and Danceability for clustering visualization
    if 'Energy' in filtered_df.columns and 'Danceability' in filtered_df.columns:
        # Sample data for better performance if dataset is large
        if len(filtered_df) > 1000:
            plot_df = filtered_df.sample(1000, random_state=42)
        else:
            plot_df = filtered_df
        
        fig = px.scatter(
            plot_df,
            x='Energy',
            y='Danceability',
            color='CDR Genre',
            title='Audio Feature Clustering by Genre',
            labels={
                'Energy': 'Energy Level', 
                'Danceability': 'Danceability Score'
            },
            size='Overall Rating',
            hover_data=['Song', 'Artist', 'Year'],
            opacity=0.7,
            size_max=15
        )
        
        fig.update_layout(
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font_color='white',
            height=400,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            )
        )
        
        # Update marker styles for better visibility
        fig.update_traces(
            marker=dict(line=dict(width=0.5, color='DarkSlateGrey')),
            selector=dict(mode='markers')
        )
        
    else:
        return create_empty_figure("Energy and Danceability features not available")
    
    return fig

@callback(
    Output("genre-audio-chart", "figure"),
    Input('filter-store', 'data'),
    Input('year-range-slider-prediction', 'value')
)
def update_genre_audio_profiles(filter_data, year_range):
    """Update genre audio profiles visualization"""
    filtered_df = apply_filters(df, filter_data, year_range)
    
    if len(filtered_df) == 0:
        return create_empty_figure("No data available for selected filters")
    
    # Get top genres and their audio profiles
    top_genres = filtered_df['CDR Genre'].value_counts().head(6).index
    audio_features = ['Energy', 'Danceability', 'BPM', 'Loudness (dB)']
    
    # Ensure features exist in dataframe
    available_features = [f for f in audio_features if f in filtered_df.columns]
    
    if len(available_features) == 0 or len(top_genres) == 0:
        return create_empty_figure("Insufficient data for genre analysis")
    
    # Calculate average audio features by genre
    genre_audio = filtered_df[filtered_df['CDR Genre'].isin(top_genres)].groupby('CDR Genre')[available_features].mean().reset_index()
    
    # Create grouped bar chart for genre audio profiles
    fig = go.Figure()
    
    colors_list = ['#EF553B', '#00CC96', '#636EFA', '#AB63FA', '#FFA15A', '#19D3F3']
    
    for i, feature in enumerate(available_features):
        fig.add_trace(go.Bar(
            name=feature,
            x=genre_audio['CDR Genre'],
            y=genre_audio[feature],
            marker_color=colors_list[i % len(colors_list)],
            text=genre_audio[feature].round(2),
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Genre Audio Profiles - Feature Comparison',
        xaxis_title='Genre',
        yaxis_title='Feature Value',
        barmode='group',
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig