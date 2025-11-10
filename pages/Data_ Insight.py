import pandas as pd
import dash_mantine_components as dmc
from dash import Input, Output, dcc, callback
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

# Add year range slider for this page (since it's specific to Data_Insight)
year_range_slider = dcc.RangeSlider(
    id='year-range-slider',
    min=int(df['Year'].min()),
    max=int(df['Year'].max()),
    value=[int(df['Year'].min()), int(df['Year'].max())],
    marks={str(year): str(year) for year in range(int(df['Year'].min()), int(df['Year'].max())+1, 10)},
    step=1,
    tooltip={"placement": "bottom", "always_visible": True}
)

# Create the main tabs component
component = dmc.Tabs(
    [
        dmc.TabsList(
            [
                dmc.TabsTab("Instrument Trends", value="instruments", color='teal'),
                dmc.TabsTab("Vocal Trends", value="vocal", color='orange'),
                dmc.TabsTab("Audio Features", value="audio", color='grape'),
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

# Create insights tabs component - FIXED: Added missing TabsTab entries
component1 = dmc.Tabs(
    [
        dmc.TabsList(
            [
                dmc.TabsTab("Success Correlations", value="correlations"),
                dmc.TabsTab("Genre Analysis", value="genre_analysis"),
                dmc.TabsTab("High vs Low Rated", value="rating_comparison"),
                dmc.TabsTab("Vocal Techniques", value="Vocal_Techniques"),  # ADDED THIS
                dmc.TabsTab("Instrument Prevalence", value="Instrument"),   # ADDED THIS
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
        dmc.TabsPanel(
            dcc.Graph(id="Vocal-Techniques-chart"),
            value="Vocal_Techniques"
        ),
        dmc.TabsPanel(
            dcc.Graph(id="Instrument-Prevalence-chart"),
            value="Instrument"
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
                
                # Filters Section - Only show year range slider here since other filters are in sidebar
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
                                "Use the filters in the sidebar to filter by Energy Level, Danceability, Genre, and Producers", 
                                size="sm", c="dimmed", ta="center"
                            ))
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
                                                        dcc.Graph(id="vocal-prevalence1", style={'height': '200px'})
                                                    ],
                                                    style={'backgroundColor': colors['card_bg']},
                                                    p="sm", shadow='sm', radius='md'
                                                ),
                                                dmc.Paper(
                                                    [
                                                        dmc.Title("Top Instruments", order=5, mb="sm", c='white'),
                                                        dcc.Graph(id="instrument-prevalence1", style={'height': '200px'})
                                                    ],
                                                    style={'backgroundColor': colors['card_bg']},
                                                    p="sm", shadow='sm', radius='md'
                                                )
                                            ],
                                            gap="md"
                                        )
                                    ]
                                )
                            ],gutter="md"
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

# Helper function to apply filters
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

# Callbacks for all interactive components
@callback(
    Output("instrument-trends-chart", "figure"),
    Output("vocal-trends-chart", "figure"),
    Output("audio-features-chart", "figure"),
    Input('filter-store', 'data'),
    Input('year-range-slider', 'value'),
)
def update_main_tabs(filter_data, year_range):
    filtered_df = apply_filters(df, filter_data, year_range)
    
    # Check if we have data after filtering
    if filtered_df.empty:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No data available for selected filters", 
            x=0.5, y=0.5, showarrow=False, 
            font=dict(color='white', size=14)
        )
        empty_fig.update_layout(
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font_color='white'
        )
        return empty_fig, empty_fig, empty_fig
    
    # Group by year for trends
    yearly_filtered = filtered_df.groupby('Year')[vocal_techniques + top_instruments + ['Energy', 'Danceability', 'BPM']].mean().reset_index()
    
    # Instrument Trends Chart
    fig1 = px.line(
        yearly_filtered, 
        x='Year', 
        y=top_instruments[:6],
        labels={'value': 'Percentage of Songs', 'variable': 'Instrument'},
        title='Instrument Trends Over Time'
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
    Output("vocal-prevalence1", "figure"),
    Output("instrument-prevalence1", "figure"),
    Input('filter-store', 'data'),
    Input('year-range-slider', "value"),
)
def update_prevalence_charts(filter_data, year_range):
    filtered_df = apply_filters(df, filter_data, year_range)
    
    # Check if filtered data is empty
    if filtered_df.empty:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No data available for selected filters", 
            x=0.5, y=0.5, showarrow=False, 
            font=dict(color='white', size=14)
        )
        empty_fig.update_layout(
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font_color='white',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        return empty_fig, empty_fig
    
    # Vocal Prevalence Chart
    try:
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
    except Exception as e:
        print(f"Error creating vocal chart: {e}")
        fig1 = go.Figure()
        fig1.add_annotation(text="Error creating vocal chart", x=0.5, y=0.5, showarrow=False)
    
    fig1.update_layout(
        showlegend=False, 
        margin=dict(l=50, r=20, t=20, b=20),
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white',
        xaxis=dict(showgrid=False, range=[0, 1]),
        yaxis=dict(showgrid=False)
    )
    fig1.update_traces(
        marker_line_width=0,
        hovertemplate='<b>%{y}</b><br>Percentage: %{x:.1%}<extra></extra>'
    )
    
    # Instrument Prevalence Chart
    try:
        instrument_summary = filtered_df[top_instruments].mean().sort_values(ascending=True)
        
        fig2 = px.bar(
            x=instrument_summary.values,
            y=instrument_summary.index,
            orientation='h',
            title='',
            labels={'x': 'Percentage', 'y': ''},
            color=instrument_summary.values,
            color_continuous_scale='plasma'
        )
    except Exception as e:    
        print(f"Error creating instrument chart: {e}")
        fig2 = go.Figure()
        fig2.add_annotation(text="Error creating instrument chart", x=0.5, y=0.5, showarrow=False)
    
    fig2.update_layout(
        showlegend=False, 
        margin=dict(l=50, r=20, t=20, b=20),
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg'],
        font_color='white',
        xaxis=dict(showgrid=False, range=[0, 1]),
        yaxis=dict(showgrid=False)
    )
    fig2.update_traces(
        marker_line_width=0,
        hovertemplate='<b>%{y}</b><br>Percentage: %{x:.1%}<extra></extra>'
    )
    
    return fig1, fig2

@callback(
    Output("correlation-chart", "figure"),
    Output("genre-analysis-chart", "figure"),
    Output("rating-comparison-chart", "figure"),
    Output("Vocal-Techniques-chart", "figure"),
    Output("Instrument-Prevalence-chart", "figure"),
    Input('filter-store', 'data'),
    Input('year-range-slider', 'value')
)
def update_insights_tabs(filter_data, year_range):
    filtered_df = apply_filters(df, filter_data, year_range)
    
    # Check if we have data
    if filtered_df.empty:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        empty_fig.update_layout(plot_bgcolor=colors['card_bg'], paper_bgcolor=colors['card_bg'], font_color='white')
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
    
    # Correlation Chart
    try:
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
    except Exception as e:
        print(f"Error creating correlation chart: {e}")
        fig1 = go.Figure()
        fig1.add_annotation(text="Error creating chart", x=0.5, y=0.5, showarrow=False)
        fig1.update_layout(plot_bgcolor=colors['card_bg'], paper_bgcolor=colors['card_bg'], font_color='white')
    
    # Genre Analysis
    try:
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
    except Exception as e:
        print(f"Error creating genre analysis chart: {e}")
        fig2 = go.Figure()
        fig2.add_annotation(text="Error creating chart", x=0.5, y=0.5, showarrow=False)
        fig2.update_layout(plot_bgcolor=colors['card_bg'], paper_bgcolor=colors['card_bg'], font_color='white')
    
    # Rating Comparison
    try:
        high_rated = filtered_df[filtered_df['Overall Rating'] >= 4]
        low_rated = filtered_df[filtered_df['Overall Rating'] <= 2]
        
        comparison_data = pd.DataFrame({
            'Feature': ['Energy', 'Danceability', 'BPM', 'Explicit', 'Falsetto Vocal'],
            'High Rated': [
                high_rated['Energy'].mean() if not high_rated.empty else 0,
                high_rated['Danceability'].mean() if not high_rated.empty else 0,
                high_rated['BPM'].mean() if not high_rated.empty else 0,
                high_rated['Explicit'].mean() if not high_rated.empty else 0,
                high_rated['Falsetto Vocal'].mean() if not high_rated.empty else 0
            ],
            'Low Rated': [
                low_rated['Energy'].mean() if not low_rated.empty else 0,
                low_rated['Danceability'].mean() if not low_rated.empty else 0,
                low_rated['BPM'].mean() if not low_rated.empty else 0,
                low_rated['Explicit'].mean() if not low_rated.empty else 0,
                low_rated['Falsetto Vocal'].mean() if not low_rated.empty else 0
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
    except Exception as e:
        print(f"Error creating rating comparison chart: {e}")
        fig3 = go.Figure()
        fig3.add_annotation(text="Error creating chart", x=0.5, y=0.5, showarrow=False)
        fig3.update_layout(plot_bgcolor=colors['card_bg'], paper_bgcolor=colors['card_bg'], font_color='white')

    # Vocal Techniques by Genre Heatmap - FIXED: Moved outside the rating comparison try block
    try:
        genre_vocal_analysis = filtered_df.groupby('CDR Genre')[vocal_techniques].mean()
        
        fig4 = px.imshow(
            genre_vocal_analysis.T,
            title='Vocal Techniques Prevalence by Genre',
            aspect="auto",
            color_continuous_scale='viridis'
        )
        fig4.update_layout(
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font_color='white',
            xaxis_title='Genre',
            yaxis_title='Vocal Technique'
        )
    except Exception as e:
        print(f"Error creating vocal techniques chart: {e}")
        fig4 = go.Figure()
        fig4.add_annotation(text="Error creating vocal techniques chart", x=0.5, y=0.5, showarrow=False)
        fig4.update_layout(plot_bgcolor=colors['card_bg'], paper_bgcolor=colors['card_bg'], font_color='white')

    # Instrument Prevalence by Genre Heatmap - FIXED: Moved outside the rating comparison try block
    try:
        genre_instrument_analysis = filtered_df.groupby('CDR Genre')[top_instruments].mean()
        
        fig5 = px.imshow(
            genre_instrument_analysis.T,
            title='Instrument Prevalence by Genre',
            aspect="auto",
            color_continuous_scale='plasma'
        )
        fig5.update_layout(
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font_color='white',
            xaxis_title='Genre',
            yaxis_title='Instrument'
        )
    except Exception as e:
        print(f"Error creating instrument prevalence chart: {e}")
        fig5 = go.Figure()
        fig5.add_annotation(text="Error creating instrument prevalence chart", x=0.5, y=0.5, showarrow=False)
        fig5.update_layout(plot_bgcolor=colors['card_bg'], paper_bgcolor=colors['card_bg'], font_color='white')

    return fig1, fig2, fig3, fig4, fig5

@callback(
    Output("audio-distribution-chart", "figure"),
    Output("era-analysis-chart", "figure"),
    Input('filter-store', 'data'),
    Input('year-range-slider', 'value')
)
def update_additional_charts(filter_data, year_range):
    filtered_df = apply_filters(df, filter_data, year_range)
    
    # Check if we have data
    if filtered_df.empty:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        empty_fig.update_layout(plot_bgcolor=colors['card_bg'], paper_bgcolor=colors['card_bg'], font_color='white')
        return empty_fig, empty_fig
    
    # Audio Distribution
    try:
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
    except Exception as e:
        print(f"Error creating audio distribution chart: {e}")
        fig1 = go.Figure()
        fig1.add_annotation(text="Error creating chart", x=0.5, y=0.5, showarrow=False)
        fig1.update_layout(plot_bgcolor=colors['card_bg'], paper_bgcolor=colors['card_bg'], font_color='white')
    
    # Era Analysis
    try:
        # Create era categories
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
                go.Bar(x=era_df['Era'], y=era_df['Avg Rating'], name='Rating', marker_color=colors_bars[0]), 
                row=1, col=1
            )
            fig2.add_trace(
                go.Bar(x=era_df['Era'], y=era_df['Avg Weeks at #1'], name='Weeks', marker_color=colors_bars[1]), 
                row=1, col=2
            )
            fig2.add_trace(
                go.Bar(x=era_df['Era'], y=era_df['Energy'], name='Energy', marker_color=colors_bars[2]), 
                row=2, col=1
            )
            fig2.add_trace(
                go.Bar(x=era_df['Era'], y=era_df['Danceability'], name='Danceability', marker_color=colors_bars[3]), 
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
    except Exception as e:
        print(f"Error creating era analysis chart: {e}")
        fig2 = go.Figure()
        fig2.add_annotation(text="Error creating chart", x=0.5, y=0.5, showarrow=False)
        fig2.update_layout(plot_bgcolor=colors['card_bg'], paper_bgcolor=colors['card_bg'], font_color='white')
    
    return fig1, fig2