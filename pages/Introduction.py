import dash_mantine_components as dmc
from dash import html, dcc
import dash_ag_grid as dag
from utils import df, metrics, colors
import dash



dash.register_page(
    __name__,
    title='Introduction',
    path= '/Introducton', 
    description= 'Introducton',
    order=1
)

# Create column definitions for the AG Grid
columnDefs = []

for col in df.columns:
    if col in df.select_dtypes(include='object').columns:
        columnDef = {
            'field': col, 
            'headerName': col.replace('_', ' ').title(),
            'filter': True,
            'floatingFilter': True,
            'minWidth': 150
        }
    elif col in df.select_dtypes(include=['int', 'float']).columns:
        # Special formatting for percentage columns
        if 'Pct' in col or 'pct' in col or 'Percentage' in col:
            columnDef = {
                'field': col,
                'headerName': col.replace('_', ' ').title(),
                'valueFormatter': {"function": "d3.format('.1f')(params.value) + '%'"},
                'type': 'rightAligned',
                'filter': True,
                'floatingFilter': True,
                'minWidth': 150
            }
        # Special formatting for year columns
        elif 'Year' in col or 'Date' in col:
            columnDef = {
                'field': col,
                'headerName': col.replace('_', ' ').title(),
                'valueFormatter': {"function": "d3.format('')(params.value)"},
                'type': 'rightAligned',
                'filter': True,
                'floatingFilter': True,
                'minWidth': 120
            }
        # Format rating columns
        elif 'Rating' in col:
            columnDef = {
                'field': col,
                'headerName': col.replace('_', ' ').title(),
                'valueFormatter': {"function": "d3.format('.1f')(params.value)"},
                'type': 'rightAligned',
                'filter': True,
                'floatingFilter': True,
                'minWidth': 120
            }
        # Default numeric formatting
        else:
            columnDef = {
                'field': col,
                'headerName': col.replace('_', ' ').title(),
                'valueFormatter': {"function": "d3.format('.2f')(params.value)"},
                'type': 'rightAligned',
                'filter': True,
                'floatingFilter': True,
                'minWidth': 130
            }
    else:
        # Default for other data types (datetime, bool, etc.)
        columnDef = {
            'field': col,
            'headerName': col.replace('_', ' ').title(),
            'filter': True,
            'floatingFilter': True,
            'minWidth': 150
        }
    
    columnDefs.append(columnDef)

# Customize specific important columns
for col_def in columnDefs:
    # Make key identifiers non-editable
    if col_def['field'] in ['Song', 'Artist', 'Date', 'Year']:
        col_def['editable'] = False
        col_def['pinned'] = 'left'
        col_def['minWidth'] = 180
    
    # Add specific width for long text columns
    if col_def['field'] in ['Lyrics', 'Songwriters', 'CDR Genre', 'Discogs Genre']:
        col_def['width'] = 250
    
    # Add tooltips for complex columns
    if col_def['field'] in ['CDR Genre', 'Discogs Genre', 'CDR Style', 'Discogs Style']:
        col_def['tooltipField'] = col_def['field']

# Create the grid component
grid = html.Div([
    dmc.Paper([
        dmc.Stack([
            dmc.Title('Billboard Hot 100 Dataset Explorer', order=3, c=colors['primary']),
            dmc.Text(
                "Explore the complete dataset of Billboard Hot 100 Number One songs with filtering, sorting, and search capabilities",
                c=colors['text_secondary'],
                size="md"
            ),
            dag.AgGrid(
                id='music-dataset-grid',
                columnDefs=columnDefs,
                rowData=df.to_dict('records'),
                defaultColDef={
                    'editable': True, 
                    "filter": True, 
                    "floatingFilter": True,
                    "resizable": True,
                    "sortable": True
                },
                dashGridOptions={
                    "suppressFieldDotNotation": True,
                    "pagination": True,
                    "paginationPageSize": 20,
                    "animateRows": True,
                    "rowSelection": "multiple"
                },
                columnSize="sizeToFit",
                style={"height": "600px", "width": "100%"}
            ),
            dmc.Group([
                dmc.Text("Use the filters above each column to explore the data", size="sm", c=colors['text_secondary']),
                dmc.Badge("Interactive Dataset", c="blue", variant="light")
            ], align="apart")
        ], align="lg")
    ], p="md", withBorder=True, shadow="md")
])

# Create the introduction layout
layout = dmc.Container([
    dmc.Stack([
        # Header Section
        dmc.Paper([
            dmc.Stack([
                dmc.Title('The Role of Vocal Techniques and Instrumentation in Hit Songs', 
                         c=colors['text_primary'], 
                         order=1),
                dmc.Text(
                    "Analyzing how vocal styles and musical instrumentation contribute to chart-topping success across decades",
                    c=colors['text_secondary'], 
                    size="xl",
                    tt='center'
                ),
                dmc.Divider(variant="solid", c=colors['primary']),
                dmc.Text(
                    "This comprehensive analysis reveals patterns and trends in Billboard Hot 100 Number One songs, "
                    "providing data-driven insights for music professionals, artists, and producers.",
                    c=colors['text_secondary'],
                    size="md",
                    tt='center'
                )
            ], gap="xs")
        ], p="xl", style={'background': colors['accent_gradient']}),
        
        # Key Metrics Overview Cards
        dmc.Grid([
            # Dataset Overview Cards
            dmc.GridCol(span=3, children=dmc.Card(
                children=[
                    dmc.Group([
                        dmc.Badge("📊", size="xl"),
                        dmc.Text("Total Hit Songs", size="sm", c="dimmed"),
                    ]),
                    dmc.Text(f"{metrics['total_songs']:,}", size="xl", fw=700, c=colors['primary']),
                    dmc.Text(f"{metrics['unique_artists']} Unique Artists", size="sm", c=colors['success']),
                ],
                withBorder=True,
                shadow="lg",
                radius="lg",
                style={'background': colors['card_bg']}
            )),
            
            dmc.GridCol(span=3, children=dmc.Card(
                children=[
                    dmc.Group([
                        dmc.Badge("🏆", size="xl"),
                        dmc.Text("Avg Weeks at #1", size="sm", c="dimmed"),
                    ]),
                    dmc.Text(f"{metrics['avg_weeks_at_no1']:.1f}", size="xl", fw=700, c=colors['warning']),
                    dmc.Text(f"Avg Rating: {metrics['avg_overall_rating']:.1f}/5", size="sm", c=colors['secondary']),
                ],
                withBorder=True,
                shadow="lg",
                radius="lg",
                style={'background': colors['card_bg']}
            )),
            
            dmc.GridCol(span=3, children=dmc.Card(
                children=[
                    dmc.Group([
                        dmc.Badge("⚡", size="xl"),
                        dmc.Text("Avg Energy Level", size="sm", c="dimmed"),
                    ]),
                    dmc.Text(f"{metrics['avg_energy']:.0f}%", size="xl", fw=700, c=colors['danger']),
                    dmc.Text(f"BPM: {metrics['avg_bpm']:.0f}", size="sm", c=colors['danger']),
                ],
                withBorder=True,
                shadow="lg",
                radius="lg",
                style={'background': colors['card_bg']}
            )),
            
            dmc.GridCol(span=3, children=dmc.Card(
                children=[
                    dmc.Group([
                        dmc.Badge("💃", size="xl"),
                        dmc.Text("Danceability", size="sm", c="dimmed"),
                    ]),
                    dmc.Text(f"{metrics['avg_danceability']:.0f}%", size="xl", fw=700, c=colors['success']),
                    dmc.Text("Hit Song Factor", size="sm", c=colors['text_secondary']),
                ],
                withBorder=True,
                shadow="lg",
                radius="lg",
                style={'background': colors['card_bg']}
            )),
        ], gutter="xl", style={'marginBottom': '30px'}),
        
        # Vocal Techniques Section
        dmc.Paper([
            dmc.Stack([
                dmc.Title("🎤 Vocal Techniques Analysis", order=3, c=colors['primary']),
                dmc.Text("Prevalence of different vocal styles in chart-topping songs", c=colors['text_secondary']),
                dmc.Grid([
                    dmc.GridCol(span=4, children=dmc.Card(
                        children=[
                            dmc.Text("Falsetto Usage", size="sm", c="dimmed"),
                            dmc.Text(f"{metrics['songs_with_falsetto']}", size="xl", fw=600),
                            dmc.Progress(
                                value=metrics['pct_falsetto'],
                                color="violet",
                                size="lg",
                                radius="xl"
                            ),
                            dmc.Text(f"{metrics['pct_falsetto']:.1f}% of hits", size="xs", c="dimmed"),
                        ],
                        withBorder=True,
                        shadow="sm",
                        radius="md",
                    )),
                    
                    dmc.GridCol(span=4, children=dmc.Card(
                        children=[
                            dmc.Text("Rap Verses in Pop", size="sm", c="dimmed"),
                            dmc.Text(f"{metrics['songs_with_rap_verse']}", size="xl", fw=600),
                            dmc.Progress(
                                value=metrics['pct_rap_verse'],
                                color="danger",
                                size="lg",
                                radius="xl"
                            ),
                            dmc.Text(f"{metrics['pct_rap_verse']:.1f}% of hits", size="xs", c="dimmed"),
                        ],
                        withBorder=True,
                        shadow="sm",
                        radius="md",
                    )),
                    
                    dmc.GridCol(span=4, children=dmc.Card(
                        children=[
                            dmc.Text("Vocally Based Songs", size="sm", c="dimmed"),
                            dmc.Text(f"{metrics['vocally_based_songs']}", size="xl", fw=600),
                            dmc.Progress(
                                value=metrics['pct_vocally_based'],
                                color="blue",
                                size="lg",
                                radius="xl"
                            ),
                            dmc.Text(f"{metrics['pct_vocally_based']:.1f}%", size="xs", c="dimmed"),
                        ],
                        withBorder=True,
                        shadow="sm",
                        radius="md",
                    )),
                ], gutter="lg"),
                
                dmc.Grid([
                    dmc.GridCol(span=6, children=dmc.Card(
                        children=[
                            dmc.Text("Explicit Content", size="sm", c="dimmed"),
                            dmc.Text(f"{metrics['explicit_songs']}", size="xl", fw=600, c=colors['danger']),
                            dmc.Text(f"Spoken Word: {metrics['songs_with_spoken_word']}", size="sm", c=colors['text_secondary']),
                            dmc.Text("Modern Trend Indicator", size="xs", c="dimmed"),
                        ],
                        withBorder=True,
                        shadow="sm",
                        radius="md",
                    )),
                    
                    dmc.GridCol(span=6, children=dmc.Card(
                        children=[
                            dmc.Text("Vocal Introductions", size="sm", c="dimmed"),
                            dmc.Text(f"{(df['Vocal Introduction'] == 1).sum()}", size="xl", fw=600),
                            dmc.Text(f"Free Time: {(df['Free Time Vocal Introduction'] == 1).sum()}", size="sm", c=colors['text_secondary']),
                            dmc.Text("Song Structure", size="xs", c="dimmed"),
                        ],
                        withBorder=True,
                        shadow="sm",
                        radius="md",
                    )),
                ], gutter="lg"),
            ], gap="lg")
        ], p="lg", withBorder=True, shadow="sm", style={'marginBottom': '30px'}),
        
        # Instrumentation Section
        dmc.Paper([
            dmc.Stack([
                dmc.Title("🎵 Instrumentation Patterns", order=3, c=colors['primary']),
                dmc.Text("Dominant instruments and their prevalence in hit songs", c=colors['text_secondary']),
                dmc.Grid([
                    dmc.GridCol(span=3, children=dmc.Card(
                        children=[
                            dmc.Text("🎸 Guitar-Based", size="sm", c="dimmed"),
                            dmc.Text(f"{metrics['guitar_based_songs']}", size="xl", fw=600),
                            dmc.Progress(
                                value=metrics['pct_guitar'],
                                color="green",
                                size="md",
                                radius="xl"
                            ),
                            dmc.Text(f"{metrics['pct_guitar']:.1f}%", size="xs", c="dimmed"),
                        ],
                        withBorder=True,
                        shadow="sm",
                        radius="md",
                    )),
                    
                    dmc.GridCol(span=3, children=dmc.Card(
                        children=[
                            dmc.Text("🎹 Piano/Keyboard", size="sm", c="dimmed"),
                            dmc.Text(f"{metrics['piano_based_songs']}", size="xl", fw=600),
                            dmc.Progress(
                                value=metrics['pct_piano'],
                                color="blue",
                                size="md",
                                radius="xl"
                            ),
                            dmc.Text(f"{metrics['pct_piano']:.1f}%", size="xs", c="dimmed"),
                        ],
                        withBorder=True,
                        shadow="sm",
                        radius="md",
                    )),
                    
                    dmc.GridCol(span=3, children=dmc.Card(
                        children=[
                            dmc.Text("🎻 Orchestral", size="sm", c="dimmed"),
                            dmc.Text(f"{metrics['orchestral_songs']}", size="lg", fw=600),
                            dmc.Text(f"Bass-Based: {metrics['bass_based_songs']}", size="sm", c=colors['danger']),
                            dmc.Text("Symphonic Elements", size="xs", c="dimmed"),
                        ],
                        withBorder=True,
                        shadow="sm",
                        radius="md",
                    )),
                    
                    dmc.GridCol(span=3, children=dmc.Card(
                        children=[
                            dmc.Text("🎺 Horns & Winds", size="sm", c="dimmed"),
                            dmc.Text(f"{metrics['horn_wind_songs']}", size="lg", fw=600),
                            dmc.Text("Brass Sections", size="sm", c='violet'),
                            dmc.Text("Classic Influence", size="xs", c="dimmed"),
                        ],
                        withBorder=True,
                        shadow="sm",
                        radius="md",
                    )),
                ], gutter="lg"),
            ], gap="lg")
        ], p="lg", withBorder=True, shadow="sm", style={'marginBottom': '30px'}),
        
        # Summary Insights
        dmc.Grid([
            dmc.GridCol(span=8, children=dmc.Paper([
                dmc.Stack([
                    dmc.Title("📈 Key Insights", order=3, c=colors['primary']),
                    dmc.List([
                        dmc.ListItem([
                            dmc.Text("Guitar and piano remain foundational instruments, appearing in ", span=True),
                            dmc.Text(f"{metrics['pct_guitar']:.1f}%", c=colors['success'], span=True, fw=600),
                            dmc.Text(" and ", span=True),
                            dmc.Text(f"{metrics['pct_piano']:.1f}%", c=colors['success'], span=True, fw=600),
                            dmc.Text(" of hit songs respectively", span=True),
                        ]),
                        dmc.ListItem([
                            dmc.Text("Falsetto vocals appear in ", span=True),
                            dmc.Text(f"{metrics['pct_falsetto']:.1f}%", c='violet', span=True, fw=600),
                            dmc.Text(" of chart-toppers, indicating its emotional impact", span=True),
                        ]),
                        dmc.ListItem([
                            dmc.Text("Rap elements in non-rap songs show cross-genre influence (", span=True),
                            dmc.Text(f"{metrics['pct_rap_verse']:.1f}%", c='orange', span=True, fw=600),
                            dmc.Text("), reflecting modern music fusion", span=True),
                        ]),
                        dmc.ListItem([
                            dmc.Text("Energy (", span=True),
                            dmc.Text(f"{metrics['avg_energy']:.0f}%", c=colors['danger'], span=True, fw=600),
                            dmc.Text(") and danceability (", span=True),
                            dmc.Text(f"{metrics['avg_danceability']:.0f}%", c=colors['success'], span=True, fw=600),
                            dmc.Text(") are consistent factors in hit potential", span=True),
                        ]),
                    ], spacing="xs"),
                    dmc.Text(
                        "These metrics reveal how specific vocal techniques and instrumentation choices correlate with commercial success and audience engagement across different eras.",
                        c="dimmed",
                        size="sm"
                    ),
                ])
            ], p="lg", withBorder=True)),
            
            dmc.GridCol(span=4, children=dmc.Paper([
                dmc.Stack([
                    dmc.Title("🎯 Top Genres", order=4, c=colors['primary']),
                    dmc.Stack([
                        dmc.Group([
                            dmc.Text(genre, size="sm"),
                            dmc.Badge(f"{count} songs", c="blue", variant="light")
                        ], align="apart") 
                        for genre, count in metrics['top_genres'].items()
                    ], gap="xs"),
                    dmc.Divider(),
                    dmc.Text("Most represented genres in Billboard #1 hits", size="sm", c="dimmed"),
                ])
            ], p="md", withBorder=True)),
        ], gutter="lg", style={'marginBottom': '30px'}),
        
        # Interactive Dataset Grid
        grid
        
    ], gap="xl")
], fluid=True, size="xl", style={'backgroundColor': colors['background'], 'minHeight': '100vh'})