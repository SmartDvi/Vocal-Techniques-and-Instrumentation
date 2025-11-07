import dash_mantine_components as dmc
from dash import Input, Output, Dash, State, dcc, _dash_renderer, callback
import dash
from utils import colors, df
from dash_iconify import DashIconify


_dash_renderer._set_react_version('18.2.0')

app = Dash(
    external_stylesheets=dmc.styles.ALL, 
    use_pages=True,
    
)
energy_level_dropdown = dmc.MultiSelect(
    id='energy_level_dropdown',
    label='Select Energy Level',
    data=[{'label': level, 'value': level} for level in df['Energy_level'].dropna().unique()],
    value=list(df['Energy_level'].unique()),
    clearable=True,
    searchable=True,
    style={'marginBottom': "15px"})

genre_dropdown = dmc.MultiSelect(
    id='genre_dropdown',
    label='Select Genres',
    data=[{'label': genre, 'value': genre} for genre in df['CDR Genre'].value_counts().head(10).index],
    value=df['CDR Genre'].value_counts().head(3).index.tolist(),
    clearable=True,
    searchable=True,
    style={'marginBottom': "15px"}
)


dropdown_danceability_level = dmc.MultiSelect(
                id='dropdown_danceability_level',
                label='Select danceabile Music Level',
                data=[{'label': Danceability_level, 'value': Danceability_level} for Danceability_level in df['Danceability_level'].dropna().unique()],
                value=list(df['Danceability_level'].unique())[:3],
                clearable=True,
                searchable=True,
                style={'marginBottom': "20px"}
        )

dropdown_Producers = dmc.MultiSelect(
                id='dropdown_Producer',
                label='Select Producer',
                data=[{'label': Producers, 'value': Producers} for Producers in df['Producers'].dropna().unique()],
                value=list(df['Producers'].unique())[:3],
                clearable=True,
                searchable=True,
                #style={'marginBottom': "20px"}
        )
year_dropdown = dmc.Select(
    id="year-dropdown",
    label='Select Year',
    data=[{'label': str(year), 'value': str(year)} for year in df['Year'].unique()],
    value='2021'
)


header = dmc.Paper(
    p='xs',
    mb='xl',
    style={
        'background': 'dask',
        'fontFamily': 'Inter, sans-serif',
    },
    children=[
        dmc.Stack(
           # gap='xs',
            children=[
                dmc.Title("The Role of Vocal Techniques and Instrumentation in Hit Songs",
                          order=1, c='blue'),
                dmc.Burger(id="burger-button", opened=False, hiddenFrom="md"),
                
            ],justify="flex-start",
        )
    ]

)



theme_toggle = dmc.ActionIcon(
    [
        dmc.Paper(DashIconify(icon="radix-icons:sun", width=25), darkHidden=True),
        dmc.Paper(DashIconify(icon="radix-icons:moon", width=25), lightHidden=True),
    ],
    variant="transparent",
    color="yellow",
    id="color-scheme-toggle",
    size="lg",
    ms="auto",
)


links = dmc.Stack(
    [
        dmc.Anchor(f"{page['name']}", href=page["relative_path"])
        for page in dash.page_registry.values()
        if page["module"] != "pages.not_found_404"
    ]
)

# developing the side setup inside a variable 
navbar = dcc.Loading(
    dmc.ScrollArea(

[
      dmc.Stack(
                [
                    theme_toggle,
                    links,
                    dropdown_danceability_level,
                    dropdown_Producers,
                    year_dropdown,
                    genre_dropdown,
                    energy_level_dropdown
        
                  

                ]
            )
], offsetScrollbars=True,
type='scroll',
style={'height': '70%'}
    ),
)


app_shell = dmc.AppShell(
    [
        dmc.AppShellHeader(header, px=25),
        dmc.AppShellNavbar(navbar, p=24),
        dmc.AppShellMain(dash.page_container, py=60, pr=5),
        dmc.AppShellFooter(
            [
                dmc.Group(
                    [
                        dmc.NavLink(
                            label= 'Sources Code',
                            description='double Sources',
                            leftSection=dmc.Badge(
                                "2", size="xs", variant='filled', color='orange', w=16, h=16,p=0
                            ),
                            childrenOffset=28,
                            children=[
                                dmc.NavLink(label="GitHub", href='https://github.com/SmartDvi/Air_Pollution.git'),
                                dmc.NavLink(label="PY.CAFE", href='https://py.cafe/SmartDvi/plotly-global-air-quality')
                            ]

                        )
                    ], justify='lg'
                )
            ]
        )
    ],
    header={"height": 70},
    padding="xl",
    navbar={
        "width": 275,
        "breakpoint": "md",
        "collapsed": {"mobile": True},
    },
    aside={
        "width": 200,
        "breakpoint": "xl",
        "collapsed": {"desktop": False, "mobile": True},
    },
    id="app-shell",
)

app.layout = dmc.MantineProvider(
     theme={
        'colorScheme': 'dark',
        'fontFamily': 'Inter, sans-serif',
    },
    children=[
        dcc.Store(id='ts', storage_type='local', data='light'),
        app_shell
    ],
    id="mantine-provider",
    forceColorScheme="light",
)


@callback(
    Output("app-shell", "navbar"),
    Input("burger-button", "opened"),
    State("app-shell", "navbar"),
)
def navbar_is_open(opened, navbar):
    navbar["collapsed"] = {"mobile": not opened}
    return navbar


@callback(
    Output("mantine-provider", "forceColorScheme"),
    Input("color-scheme-toggle", "n_clicks"),
    State("mantine-provider", "forceColorScheme"),
    prevent_initial_call=True,
)
def switch_theme(_, theme):
    return "dark" if theme == "light" else "light"



if __name__ == "__main__":
    app.run(debug=True, port=6070)






