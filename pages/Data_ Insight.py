import pandas as pd
import dash_mantine_components as dmc
from dash import Dash, Input, Output, dcc, callback
import dash
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import colors, df, vocal_techniques, top_instruments, instrument_columns, metrics

dash.register_page(
    __name__,
    path='/Data_Insight',
    title='Music Data Insights', 
    description='Deep insights into vocal techniques and instrumentation patterns',
    order=2
)

# Create filtered data for visualizations
yearly_trends = df.groupby('Year')[vocal_techniques + top_instruments].mean().reset_index()

# Calculate correlations for insights
success_correlations = df[top_instruments + vocal_techniques + ['Overall Rating', 'Weeks at Number One']].corr()[
    ['Overall Rating', 'Weeks at Number One']
].drop(['Overall Rating', 'Weeks at Number One'])

# Create dropdown components
dropdown_danceability_level = dmc.MultiSelect(
    id='dropdown_danceability_level',
    label='Select Danceability Level',
    data=[{'label': level, 'value': level} for level in df['Danceability_level'].dropna().unique()],
    value=list(df['Danceability_level'].unique()),
    clearable=True,
    searchable=True,
    style={'marginBottom': "15px"}
)

energy_level_dropdown = dmc.MultiSelect(
    id='energy_level_dropdown',
    label='Select Energy Level',
    data=[{'label': level, 'value': level} for level in df['Energy_level'].dropna().unique()],
    value=list(df['Energy_level'].unique()),
    clearable=True,
    searchable=True,
    style={'marginBottom': "15px"}
)

year_range_slider = dcc.RangeSlider(
    id='year-range-slider',
    min=df['Year'].min(),
    max=df['Year'].max(),
    value=[df['Year'].min(), df['Year'].max()],
    marks={str(year): str(year) for year in range(df['Year'].min(), df['Year'].max()+1, 10)},
    step=1,
    tooltip={"placement": "bottom", "always_visible": True}
)

genre_dropdown = dmc.MultiSelect(
    id='genre_dropdown',
    label='Select Genres',
    data=[{'label': genre, 'value': genre} for genre in df['CDR Genre'].value_counts().head(10).index],
    value=df['CDR Genre'].value_counts().head(3).index.tolist(),
    clearable=True,
    searchable=True,
    style={'marginBottom': "15px"}
)

# Create the main tabs component
component = dmc.Tabs(
    [
        dmc.TabsList(
            [
                dmc.TabsTab("Instrument Trends", value="instruments"),
                dmc.TabsTab("Vocal Trends", value="vocal"),
                dmc.TabsTab("Audio Features", value="audio"),
            ]
        ),
        dmc.TabsPanel(
            dcc.Graph(id="instrument-trends-chart"),
            value="instruments"
        ),
        dmc.TabsPanel(
            dcc.Graph(id="vocal-trends-chart"),
            value="vocal"
        ),
        dmc.TabsPanel(
            dcc.Graph(id="audio-features-chart"),
            value="audio"
        ),
    ],
    value="instruments",
    id="main-tabs"
)

# Create insights tabs component
component1 = dmc.Tabs(
    [
        dmc.TabsList(
            [
                dmc.TabsTab("Success Correlations", value="correlations"),
                dmc.TabsTab("Genre Analysis", value="genre_analysis"),
                dmc.TabsTab("High vs Low Rated", value="rating_comparison"),
            ]
        ),
        dmc.TabsPanel(
            dcc.Graph(id="correlation-chart"),
            value="correlations"
        ),
        dmc.TabsPanel(
            dcc.Graph(id="genre-analysis-chart"),
            value="genre_analysis"
        ),
        dmc.TabsPanel(
            dcc.Graph(id="rating-comparison-chart"),
            value="rating_comparison"
        ),
    ],
    value="correlations",
    id="insights-tabs"
)

layout = dmc.MantineProvider(
    children=[
        dmc.Container(
            fluid=True,
            px='xl',
            children=[
                dmc.Text('Unveiling Data Insights', tt="uppercase", size="xl", c="blue", 
                        ta="center", td="underline", fw=700, p=20),
                
                # Filters Section
                dmc.Paper(
                    p="md",
                    m='md',
                    shadow='sm',
                    children=[
                        dmc.Grid([
                            dmc.GridCol(span=3, children=dropdown_danceability_level),
                            dmc.GridCol(span=3, children=energy_level_dropdown),
                            dmc.GridCol(span=3, children=genre_dropdown),
                            dmc.GridCol(span=3, children=dmc.Text("Year Range:", size="sm")),
                        ]),
                        dmc.Grid([
                            dmc.GridCol(span=12, children=year_range_slider)
                        ])
                    ],
                    style={'backgroundColor': colors['card_bg']}
                ),
                
                dmc.Paper(
                    p=0,
                    m='md',
                    shadow='md',
                    children=[
                        # First row - Main analysis charts
                        dmc.Grid(
                            children=[
                                dmc.GridCol(
                                    span=8,
                                    children=[
                                        dmc.Paper(
                                            [
                                                dmc.Title('Evolution of Music Elements Over Time', order=4, mb="md", c='white'),
                                                component
                                            ],
                                            style={'backgroundColor': colors['card_bg']},
                                            p="md", shadow='sm', radius='md'
                                        )
                                    ]
                                ),
                                dmc.GridCol(
                                    span=4,
                                    children=[
                                        dmc.Stack(
                                            [
                                                dmc.Paper(
                                                    [
                                                        dmc.Title("Vocal Techniques Prevalence", order=5, mb="sm", c='white'),
                                                        dcc.Graph(id="vocal-prevalence-chart", style={'height': '200px'})
                                                    ],
                                                    style={'backgroundColor': colors['card_bg']},
                                                    p="sm", shadow='sm', radius='md'
                                                ),
                                                dmc.Paper(
                                                    [
                                                        dmc.Title("Top Instruments", order=5, mb="sm", c='white'),
                                                        dcc.Graph(id="instrument-prevalence-chart", style={'height': '200px'})
                                                    ],
                                                    style={'backgroundColor': colors['card_bg']},
                                                    p="sm", shadow='sm', radius='md'
                                                )
                                            ],
                                            gap="md"
                                        )
                                    ]
                                )
                            ]
                        ),
                        
                        # Second row - Insights and correlations
                        dmc.Grid(
                            children=[
                                dmc.GridCol(
                                    span=12,
                                    children=[
                                        dmc.Paper(
                                            [
                                                dmc.Title('Deep Insights & Correlations', order=4, mb="md", c='white'),
                                                component1
                                            ],
                                            style={'backgroundColor': colors['card_bg']},
                                            p="md", shadow='sm', radius='md'
                                        )
                                    ]
                                )
                            ]
                        ),
                        
                        # Third row - Additional metrics
                        dmc.Grid(
                            children=[
                                dmc.GridCol(
                                    span=6,
                                    children=[
                                        dmc.Paper(
                                            [
                                                dmc.Title('Audio Features Distribution', order=5, mb="md", c='white'),
                                                dcc.Graph(id='audio-distribution-chart', style={'height': '300px'})
                                            ],
                                            style={'backgroundColor': colors['card_bg']},
                                            p="md", shadow='sm', radius='md'
                                        )
                                    ]
                                ),
                                dmc.GridCol(
                                    span=6,
                                    children=[
                                        dmc.Paper(
                                            [
                                                dmc.Title('Success Metrics by Era', order=5, mb="md", c='white'),
                                                dcc.Graph(id='era-analysis-chart', style={'height': '300px'})
                                            ],
                                            style={'backgroundColor': colors['card_bg']},
                                            p="md", shadow='sm', radius='md'
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

# Callbacks for all interactive components
@callback(
    Output("instrument-trends-chart", "figure"),
    Output("vocal-trends-chart", "figure"),
    Output("audio-features-chart", "figure"),
    Input('year-range-slider', 'value'),
    Input('genre_dropdown', 'value')
)
def update_main_tabs(year_range, selected_genres):
    filtered_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df['CDR Genre'].isin(selected_genres)]
    
    yearly_filtered = filtered_df.groupby('Year')[vocal_techniques + top_instruments + ['Energy', 'Danceability', 'BPM']].mean().reset_index()
    
    # Instrument Trends Chart
    fig1 = px.line(
        yearly_filtered, 
        x='Year', 
        y=top_instruments[:6],
        title='Top Instrument Usage Over Time',
        labels={'value': 'Percentage of Songs', 'variable': 'Instrument'}
    )
    fig1.update_layout(
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Vocal Trends Chart
    fig2 = px.line(
        yearly_filtered, 
        x='Year', 
        y=vocal_techniques[:6],
        title='Vocal Techniques Evolution',
        labels={'value': 'Percentage of Songs', 'variable': 'Technique'}
    )
    fig2.update_layout(
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Audio Features Chart
    fig3 = make_subplots(
        rows=3, cols=1, 
        subplot_titles=('Energy', 'Danceability', 'BPM'),
        vertical_spacing=0.1
    )
    
    fig3.add_trace(
        go.Scatter(x=yearly_filtered['Year'], y=yearly_filtered['Energy'], 
                  name='Energy', line=dict(color='#EF553B')), 
        row=1, col=1
    )
    fig3.add_trace(
        go.Scatter(x=yearly_filtered['Year'], y=yearly_filtered['Danceability'], 
                  name='Danceability', line=dict(color='#00CC96')), 
        row=2, col=1
    )
    fig3.add_trace(
        go.Scatter(x=yearly_filtered['Year'], y=yearly_filtered['BPM'], 
                  name='BPM', line=dict(color='#636EFA')), 
        row=3, col=1
    )
    
    fig3.update_layout(
        height=400, 
        showlegend=False, 
        title_text="Audio Features Over Time",
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white'
    )
    
    return fig1, fig2, fig3

@callback(
    Output("vocal-prevalence-chart", "figure"),
    Output("instrument-prevalence-chart", "figure"),
    Input('year-range-slider', 'value'),
    Input('dropdown_danceability_level', 'value')
)
def update_prevalence_charts(year_range, dance_levels):
    filtered_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
    
    if dance_levels:
        filtered_df = filtered_df[filtered_df['Danceability_level'].isin(dance_levels)]
    
    # Vocal Prevalence
    vocal_summary = filtered_df[vocal_techniques].mean().sort_values(ascending=True)
    fig1 = px.bar(
        x=vocal_summary.values,
        y=vocal_summary.index,
        orientation='h',
        title='',
        labels={'x': 'Percentage', 'y': ''},
        color=vocal_summary.values,
        color_continuous_scale='viridis'
    )
    fig1.update_layout(
        showlegend=False, 
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white'
    )
    
    # Instrument Prevalence
    instrument_summary = filtered_df[top_instruments[:8]].mean().sort_values(ascending=True)
    fig2 = px.bar(
        x=instrument_summary.values,
        y=instrument_summary.index,
        orientation='h',
        title='',
        labels={'x': 'Percentage', 'y': ''},
        color=instrument_summary.values,
        color_continuous_scale='plasma'
    )
    fig2.update_layout(
        showlegend=False, 
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white'
    )
    
    return fig1, fig2

@callback(
    Output("correlation-chart", "figure"),
    Output("genre-analysis-chart", "figure"),
    Output("rating-comparison-chart", "figure"),
    Input('year-range-slider', 'value')
)
def update_insights_tabs(year_range):
    filtered_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
    
    # Correlation Chart
    correlations = filtered_df[top_instruments + vocal_techniques + ['Overall Rating']].corr()['Overall Rating'].drop('Overall Rating').sort_values()
    fig1 = px.bar(
        x=correlations.values,
        y=correlations.index,
        orientation='h',
        title='Correlation with Overall Rating',
        labels={'x': 'Correlation Coefficient', 'y': ''},
        color=correlations.values,
        color_continuous_scale='RdBu_r',
        range_color=[-0.3, 0.3]
    )
    fig1.update_layout(
        showlegend=False,
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white'
    )
    
    # Genre Analysis
    top_genres = filtered_df['CDR Genre'].value_counts().head(6).index
    genre_analysis = filtered_df[filtered_df['CDR Genre'].isin(top_genres)].groupby('CDR Genre')[['Energy', 'Danceability', 'Overall Rating']].mean()
    
    fig2 = px.scatter(
        genre_analysis,
        x='Energy',
        y='Danceability',
        size='Overall Rating',
        color=genre_analysis.index,
        title='Genre Characteristics',
        size_max=30
    )
    fig2.update_layout(
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white'
    )
    
    # Rating Comparison
    high_rated = filtered_df[filtered_df['Overall Rating'] >= 4]
    low_rated = filtered_df[filtered_df['Overall Rating'] <= 2]
    
    comparison_data = pd.DataFrame({
        'Feature': ['Energy', 'Danceability', 'BPM', 'Explicit', 'Falsetto Vocal'],
        'High Rated': [
            high_rated['Energy'].mean(),
            high_rated['Danceability'].mean(),
            high_rated['BPM'].mean(),
            high_rated['Explicit'].mean(),
            high_rated['Falsetto Vocal'].mean()
        ],
        'Low Rated': [
            low_rated['Energy'].mean(),
            low_rated['Danceability'].mean(),
            low_rated['BPM'].mean(),
            low_rated['Explicit'].mean(),
            low_rated['Falsetto Vocal'].mean()
        ]
    })
    
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name='High Rated', 
        x=comparison_data['Feature'], 
        y=comparison_data['High Rated'],
        marker_color='#00CC96'
    ))
    fig3.add_trace(go.Bar(
        name='Low Rated', 
        x=comparison_data['Feature'], 
        y=comparison_data['Low Rated'],
        marker_color='#EF553B'
    ))
    fig3.update_layout(
        title='High vs Low Rated Songs Comparison', 
        barmode='group',
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white'
    )
    
    return fig1, fig2, fig3

@callback(
    Output("audio-distribution-chart", "figure"),
    Output("era-analysis-chart", "figure"),
    Input('year-range-slider', 'value')
)
def update_additional_charts(year_range):
    filtered_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
    
    # Audio Distribution
    fig1 = go.Figure()
    fig1.add_trace(go.Box(y=filtered_df['Energy'], name='Energy', boxpoints='outliers', marker_color='#EF553B'))
    fig1.add_trace(go.Box(y=filtered_df['Danceability'], name='Danceability', boxpoints='outliers', marker_color='#00CC96'))
    fig1.add_trace(go.Box(y=filtered_df['BPM'], name='BPM', boxpoints='outliers', marker_color='#636EFA'))
    fig1.update_layout(
        title='Audio Features Distribution',
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white'
    )
    
    # Era Analysis
    # Create era categories if not already in dataframe
    if 'Era' not in df.columns:
        current_year = pd.to_datetime('today').year
        filtered_df['Era'] = pd.cut(
            filtered_df['Year'], 
            bins=[1950, 1970, 1990, 2010, current_year],
            labels=['50s-60s', '70s-80s', '90s-00s', '10s-Present']
        )
    
    era_data = []
    for era in filtered_df['Era'].cat.categories:
        era_df = filtered_df[filtered_df['Era'] == era]
        if not era_df.empty:
            era_data.append({
                'Era': era,
                'Avg Rating': era_df['Overall Rating'].mean(),
                'Avg Weeks at #1': era_df['Weeks at Number One'].mean(),
                'Energy': era_df['Energy'].mean(),
                'Danceability': era_df['Danceability'].mean()
            })
    
    era_df = pd.DataFrame(era_data)
    
    fig2 = make_subplots(
        rows=2, cols=2, 
        subplot_titles=('Avg Rating', 'Weeks at #1', 'Energy', 'Danceability'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    if not era_df.empty:
        colors_bars = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
        
        fig2.add_trace(
            go.Bar(x=era_df['Era'], y=era_df['Avg Rating'], name='Rating', marker_color=colors_bars), 
            row=1, col=1
        )
        fig2.add_trace(
            go.Bar(x=era_df['Era'], y=era_df['Avg Weeks at #1'], name='Weeks', marker_color=colors_bars), 
            row=1, col=2
        )
        fig2.add_trace(
            go.Bar(x=era_df['Era'], y=era_df['Energy'], name='Energy', marker_color=colors_bars), 
            row=2, col=1
        )
        fig2.add_trace(
            go.Bar(x=era_df['Era'], y=era_df['Danceability'], name='Danceability', marker_color=colors_bars), 
            row=2, col=2
        )
    
    fig2.update_layout(
        height=400, 
        showlegend=False, 
        title_text="Music Trends by Era",
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white'
    )
    
    return fig1, fig2