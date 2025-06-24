import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Load and filter by date
CombinedHarvestSchedule_df = pd.read_csv("CMS.csv", header=0)
CombinedHarvestSchedule_df['Actual_Harvest_Date'] = pd.to_datetime(CombinedHarvestSchedule_df['Actual_Harvest_Date'])
cutoff_date = pd.to_datetime('2024-03-01')
CombinedHarvestSchedule_df = CombinedHarvestSchedule_df[CombinedHarvestSchedule_df['Actual_Harvest_Date'] > cutoff_date]

# Select relevant columns
HarvestData = CombinedHarvestSchedule_df[[
    'SiteRoom', 'Site', 'Room', 'Est__Cut_Date', 'Actual_Cut_Date',
    'Est__Replant_Date', 'Actual_Replant_Date', 'Est__Flower_Date',
    'Actual_Flower_Date', 'Est__Harvest_Date', 'Actual_Harvest_Date',
    'Est__Dry_Date', 'Actual_Dry_Date', 'Est__Ship_Avail__Date',
    'Actual_Ship_Avail__Date', 'Est__FGI_Date', 'Actual_Trim_Completion_Date',
    'Turn_Status', 'Brand', 'Strain_Commercialization_Status',
    'Strain_Accession_Number', 'Strain_Name', 'Strain_Code', 'No__Plants',
    'Canopy_Alloc_____', 'Canopy_Alloc___ft_2_', 'Est__Yield__g_ft_2_',
    'Turn_Round', 'Turn_No_', 'Harvest_Batch_No_', 'Est__Wet_Weight__lbs_',
    'Est__LTB__lbs_', 'Est__STB__lbs_', 'Est__T__lbs_', 'Est__MTB__lbs_',
    'Est__FF__lbs_', 'Est__Ship__Avail__Month', 'Act__Wet_Weight__lbs_',
    'Act__LTB__lbs_', 'Act__STB__lbs_', 'Act__T__lbs_', 'Act_MTB__lbs_',
    'Act__FF__lbs_', 'Act__Ship__Avail__Month', 'Adj__LTB__lbs_',
    'Adj__STB__lbs_', 'Adj__T__lbs_', 'Adj__MTB__lbs_', 'Adj__FF__lbs_',
    'Performance_Notes'
]].copy()

# Compute Act. LTB Eq.
HarvestData['Act_LTB_Eq'] = (
    HarvestData['Act__LTB__lbs_'].fillna(0) +
    HarvestData['Act_MTB__lbs_'].fillna(0) +
    (HarvestData['Act__FF__lbs_'].fillna(0) / 7.2)
)

# Compute Adj. LTB Eq.
HarvestData['Adj_LTB_Eq'] = (
    HarvestData['Adj__LTB__lbs_'].fillna(0) +
    HarvestData['Adj__MTB__lbs_'].fillna(0) +
    (HarvestData['Adj__FF__lbs_'].fillna(0) / 7.2)
)

# Compute QC Fails
HarvestData['QC Fails'] = HarvestData['Act_LTB_Eq'] - HarvestData['Adj_LTB_Eq']

# Yield calculations
HarvestData['LTB Yield'] = (
    HarvestData['Act_LTB_Eq'] /
    HarvestData['Canopy_Alloc___ft_2_'].replace(0, pd.NA)
) * 453.592

HarvestData['LTB/Plant'] = (
    HarvestData['Act_LTB_Eq'] /
    HarvestData['No__Plants'].replace(0, pd.NA)
) * 453.592

# Mark rows with trim complete
HarvestData['Trim_Done'] = HarvestData['Actual_Trim_Completion_Date'].notna()

# Filter to only completed trims
trimmed_df = HarvestData[HarvestData['Trim_Done'] == True].copy()

#Pull in QC form responses

def processQCFormResponses(source):
    source.rename(columns={'Batch ID':'Harvest_Batch_No_'}, inplace=True)
    source = source[['Timestamp', 'Strain Name', 'Harvest_Batch_No_',
           'Batch Size (Total Weight in Lbs)',
           'Sample Size (Sample Weight in Grams)', 'THC Percentage (% Total THC)',
           'Terpene Percentage (% Total Terpenes)', 'Microbial (Pass / Fail)',
           'Moisture Percentage Average\n(% Moisture)',
           'Trim - Category 1 Weight (g)', 'Trim - Category 2 Weight (g)',
           'Trim - Category 3 Weight (g)', 'Structure - Category 1 Weight (g)',
           'Structure - Category 2 Weight (g)',
           'Structure - Category 3 Weight (g)', 'Color - Category 1 Weight (g)',
           'Color - Category 2 Weight (g)', 'Color - Category 3 Weight (g)',
           'Size - Category 1 - Weight (g)', 'Size - Category 2 - Weight (g)',
           'Size - Category 3 - Weight (g)', 'Size - Category 4 - Weight (g)',
           'Size - Category 5 - Weight (g)', 'Size - Category 1 - Count',
           'Size - Category 2 - Count', 'Size - Category 3 - Count',
           'Size - Category 4 - Count', 'Size - Category 5 - Count']]
    df = source.copy()
    df = df.astype({'Trim - Category 1 Weight (g)': float, 'Trim - Category 2 Weight (g)': float, 'Trim - Category 3 Weight (g)': float, 'Size - Category 1 - Weight (g)': float, 'Size - Category 2 - Weight (g)': float,'Size - Category 3 - Weight (g)': float, 'Size - Category 4 - Weight (g)': float,'Size - Category 5 - Weight (g)': float, 'THC Percentage (% Total THC)':float, 'Terpene Percentage (% Total Terpenes)':float,'Moisture Percentage Average\n(% Moisture)':float})
    df['Trim Total'] = df['Trim - Category 1 Weight (g)'] + df['Trim - Category 2 Weight (g)'] + df['Trim - Category 3 Weight (g)']
    df['Trim - Category 1 Weight (g)'] = df['Trim - Category 1 Weight (g)'] / df['Trim Total']
    df['Trim - Category 2 Weight (g)'] = df['Trim - Category 2 Weight (g)'] / df['Trim Total']
    df['Trim - Category 3 Weight (g)'] = df['Trim - Category 3 Weight (g)'] / df['Trim Total']

    df['Size Total'] = df['Size - Category 1 - Weight (g)'] + df['Size - Category 2 - Weight (g)'] + df['Size - Category 3 - Weight (g)'] + df['Size - Category 4 - Weight (g)'] + df['Size - Category 5 - Weight (g)']
    df['Size - Category 1 - Weight (g)'] = df['Size - Category 1 - Weight (g)'] / df['Size Total']
    df['Size - Category 2 - Weight (g)'] = df['Size - Category 2 - Weight (g)'] / df['Size Total']
    df['Size - Category 3 - Weight (g)'] = df['Size - Category 3 - Weight (g)'] / df['Size Total']
    df['Size - Category 4 - Weight (g)'] = df['Size - Category 4 - Weight (g)'] / df['Size Total']
    df['Size - Category 5 - Weight (g)'] = df['Size - Category 5 - Weight (g)'] / df['Size Total']
    df = df[['Timestamp', 'Strain Name', 'Harvest_Batch_No_','THC Percentage (% Total THC)','Terpene Percentage (% Total Terpenes)','Moisture Percentage Average\n(% Moisture)']]
    return df

#Pull and Clean QC Data
df = pd.read_csv("QCresponses.csv", header=0)
df.replace('na',0, inplace=True)
df.replace('NA',0, inplace=True)
df = processQCFormResponses(df)
df['Harvest_Batch_No_'] = df['Harvest_Batch_No_'].astype(str).str.strip()
trimmed_df['Harvest_Batch_No_'] = trimmed_df['Harvest_Batch_No_'].astype(str).str.strip()

#merge to original dataframe baed on Harvest Batch No. or Batch ID
HarvestData = pd.merge(df, trimmed_df, on="Harvest_Batch_No_", how="outer")

print(HarvestData[['Harvest_Batch_No_', 'Strain_Name', 'THC Percentage (% Total THC)', 'Terpene Percentage (% Total Terpenes)']].dropna())
print("QC batch IDs in df:", df['Harvest_Batch_No_'].unique()[:5])
print("Trimmed batch IDs:", trimmed_df['Harvest_Batch_No_'].unique()[:5])


import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output

# Assume HarvestData is already defined
site_options = [{'label': site, 'value': site} for site in sorted(HarvestData['Site'].dropna().unique())]
brand_options = [{'label': brand, 'value': brand} for brand in sorted(HarvestData['Brand'].dropna().unique())]

app = Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        html.H2(
            "Average LTB Yield vs LTB per Plant by Strain and Brand",
            style={
                'textAlign': 'center',
                'fontWeight': 'bold',
                'fontFamily': 'Futura',
                'marginBottom': '20px'
            }
        ),

        html.Div(
            [
                dcc.DatePickerRange(
                    id='date-range',
                    min_date_allowed=HarvestData['Actual_Harvest_Date'].min(),
                    max_date_allowed=HarvestData['Actual_Harvest_Date'].max(),
                    start_date=HarvestData['Actual_Harvest_Date'].min(),
                    end_date=HarvestData['Actual_Harvest_Date'].max(),
                    display_format='YYYY-MM-DD',
                    style={'width': '100%', 'maxWidth': '350px', 'marginBottom': '10px'}
                ),
                dcc.Dropdown(
                    id='site-filter',
                    options=site_options,
                    multi=True,
                    placeholder="Filter by Site",
                    style={'width': '100%', 'maxWidth': '350px', 'marginBottom': '10px'}
                ),
                dcc.Dropdown(
                    id='brand-filter',
                    options=brand_options,
                    multi=True,
                    placeholder="Filter by Brand",
                    style={'width': '100%', 'maxWidth': '350px', 'marginBottom': '20px'}
                ),
            ],
            style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'gap': '20px',
                'justifyContent': 'center',
                'marginBottom': '20px'
            }
        ),

        html.Div(
            dcc.Graph(
                id='ltb-yield-scatter',
                config={'responsive': True},
                style={'width': '100%', 'height': '100%'}
            ),
            style={'flex': '1 1 auto', 'minHeight': 0}
        )
    ],
    style={
        'display': 'flex',
        'flexDirection': 'column',
        'height': '100vh',
        'padding': '10px',
        'boxSizing': 'border-box'
    }
)

@app.callback(
    Output('ltb-yield-scatter', 'figure'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('site-filter', 'value'),
    Input('brand-filter', 'value'),
)
def update_graph(start_date, end_date, selected_sites, selected_brands):
    if selected_sites is None:
        selected_sites = []
    if selected_brands is None:
        selected_brands = []

    df = HarvestData.dropna(subset=['Site', 'Brand', 'Actual_Harvest_Date'])

    filtered = df[
        (df['Trim_Done'] == True) &
        (df['Actual_Harvest_Date'] >= pd.to_datetime(start_date)) &
        (df['Actual_Harvest_Date'] <= pd.to_datetime(end_date))
    ]

    if selected_sites:
        filtered = filtered[filtered['Site'].isin(selected_sites)]
    if selected_brands:
        filtered = filtered[filtered['Brand'].isin(selected_brands)]


    # Define aggregation function for range excluding 0 and NaN
    def safe_range(series):
        clean = pd.to_numeric(series, errors='coerce')  # ensure numeric
        clean = clean[(clean > 0) & (~clean.isna())]    # ignore 0s and NaNs
        #print("Safe range input:", clean.tolist())      # debug
        if clean.empty:
            return "N/A"
        return f"{clean.min():.1f}–{clean.max():.1f}%"




    grouped = filtered.groupby(['Strain_Name', 'Brand']).agg(
        LTB_Yield_Mean=('LTB Yield', lambda x: x[x > 0].mean()),
        LTB_per_Plant_Mean=('LTB/Plant', lambda x: x[x > 0].mean()),
        LTB_Yield_SE=('LTB Yield', lambda x: x[x > 0].std() / np.sqrt(len(x[x > 0])) if len(x[x > 0]) > 1 else 0),
        Count=('LTB Yield', lambda x: x[x > 0].count()),
        QC_Fails=('QC Fails', lambda x: round(x.sum(), 1)),
        THC_Range=('THC Percentage (% Total THC)', safe_range),
        Terpene_Range=('Terpene Percentage (% Total Terpenes)', safe_range),
    ).reset_index()

    print(filtered[['Strain_Name', 'Brand', 'Harvest_Batch_No_','THC Percentage (% Total THC)', 'LTB Yield']].dropna().sort_values('THC Percentage (% Total THC)', ascending=False))


    fig = go.Figure()

    for brand in grouped['Brand'].unique():
        brand_data = grouped[grouped['Brand'] == brand]

        fig.add_trace(go.Scatter(
            x=brand_data['LTB_per_Plant_Mean'],
            y=brand_data['LTB_Yield_Mean'],
            error_y=dict(type='data', array=brand_data['LTB_Yield_SE']),
            mode='markers+text',
            text=brand_data['Strain_Name'],
            textposition='top center',
            name=str(brand),
            marker=dict(size=12),
            customdata=np.stack([
                brand_data['Count'],
                brand_data['QC_Fails'],
                brand_data['THC_Range'],
                brand_data['Terpene_Range']
            ], axis=-1),
            hovertemplate=(
                "<b>Strain:</b> %{text}<br>" +
                "<b>Brand:</b> " + brand + "<br>" +
                "<b>Avg LTB/Plant:</b> %{x:.2f} g<br>" +
                "<b>Avg LTB Yield:</b> %{y:.2f} g/ft²<br>" +
                "<b>Observations:</b> %{customdata[0]}<br>" +
                "<b>Total QC Fails:</b> %{customdata[1]} lbs<br>" +
                "<b>THC Range:</b> %{customdata[2]}<br>" +
                "<b>Terpene Range:</b> %{customdata[3]}"
            )
        ))

    fig.update_layout(
        title=dict(text="", x=0.5, xanchor='center', font=dict(family="Futura", size=20)),
        font=dict(family="Futura", size=12),
        xaxis=dict(title="Average LTB per Plant (g)", tickfont=dict(size=11), gridcolor='lightgray'),
        yaxis=dict(title="Average LTB Yield (g/ft²)", tickfont=dict(size=11), gridcolor='lightgray'),
        hovermode='closest',
        margin=dict(l=40, r=40, t=10, b=40),
        autosize=True
    )

    return fig

    fig.write_html("index.html", full_html=True, auto_open=False, include_plotlyjs='cdn')

if __name__ == '__main__':
    app.run()



"""
# Count observations per strain-brand
obs_count = trimmed_df.groupby(['Strain_Name', 'Brand']).size()

# Apply conditional filtering based on number of observations
rows_to_keep = []
for (strain, brand), count in obs_count.items():
    group_rows = trimmed_df[
        (trimmed_df['Strain_Name'] == strain) &
        (trimmed_df['Brand'] == brand)
    ]
    if count > 3:
        # Keep only rows with No__Plants >= 9
        filtered_group_rows = group_rows[group_rows['No__Plants'] >= 9]
    else:
        # Keep all rows
        filtered_group_rows = group_rows
    rows_to_keep.append(filtered_group_rows)

# Combine final filtered rows
filtered_df = pd.concat(rows_to_keep)

# Define RMSE
def rmse(x):
    return np.sqrt(np.mean((x - x.mean()) ** 2))

# Group by strain and brand to summarize
strain_brand_summary_df = (
    filtered_df
    .groupby(['Strain_Name', 'Brand'])
    .agg(
        LTB_Yield_Mean=('LTB Yield', 'mean'),
        LTB_Yield_Min=('LTB Yield', 'min'),
        LTB_Yield_Max=('LTB Yield', 'max'),
        LTB_Yield_Std=('LTB Yield', 'std'),
        LTB_Yield_Count=('LTB Yield', 'count'),
        LTB_Yield_RMSE=('LTB Yield', rmse),

        LTB_per_Plant_Mean=('LTB/Plant', 'mean'),
        LTB_per_Plant_Min=('LTB/Plant', 'min'),
        LTB_per_Plant_Max=('LTB/Plant', 'max'),
        LTB_per_Plant_Std=('LTB/Plant', 'std'),
        LTB_per_Plant_Count=('LTB/Plant', 'count'),
        LTB_per_Plant_RMSE=('LTB/Plant', rmse),

        QC_Fails_Mean=('QC Fails', 'mean'),
        QC_Fails_Std=('QC Fails', 'std'),
        QC_Fails_Count=('QC Fails', 'count'),
    )
    .reset_index()
)

# Round final summary for readability
strain_brand_summary_df = strain_brand_summary_df.round(2)


import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output

# (Assuming HarvestData, site_options, brand_options are already defined above)
site_options = [{'label': site, 'value': site} for site in sorted(HarvestData['Site'].dropna().unique())]
brand_options = [{'label': brand, 'value': brand} for brand in sorted(HarvestData['Brand'].dropna().unique())]

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H2(
            "Average LTB Yield vs LTB per Plant by Strain and Brand",
            style={
                'textAlign': 'center',
                'fontWeight': 'bold',
                'fontFamily': 'Futura',
                'marginBottom': '20px'
            }
        ),

        html.Div(
            [
                dcc.DatePickerRange(
                    id='date-range',
                    min_date_allowed=HarvestData['Actual_Harvest_Date'].min(),
                    max_date_allowed=HarvestData['Actual_Harvest_Date'].max(),
                    start_date=HarvestData['Actual_Harvest_Date'].min(),
                    end_date=HarvestData['Actual_Harvest_Date'].max(),
                    display_format='YYYY-MM-DD',
                    style={'width': '100%', 'maxWidth': '350px', 'marginBottom': '10px'}
                ),
                dcc.Dropdown(
                    id='site-filter',
                    options=site_options,
                    multi=True,
                    placeholder="Filter by Site",
                    style={'width': '100%', 'maxWidth': '350px', 'marginBottom': '10px'}
                ),
                dcc.Dropdown(
                    id='brand-filter',
                    options=brand_options,
                    multi=True,
                    placeholder="Filter by Brand",
                    style={'width': '100%', 'maxWidth': '350px', 'marginBottom': '20px'}
                ),
            ],
            style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'gap': '20px',
                'justifyContent': 'center',
                'marginBottom': '20px'
            }
        ),

        html.Div(
            dcc.Graph(
                id='ltb-yield-scatter',
                config={'responsive': True},
                style={'width': '100%', 'height': '100%'}
            ),
            style={'flex': '1 1 auto', 'minHeight': 0}
        )
    ],
    style={
        'display': 'flex',
        'flexDirection': 'column',
        'height': '100vh',
        'padding': '10px',
        'boxSizing': 'border-box'
    }
)

@app.callback(
    Output('ltb-yield-scatter', 'figure'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('site-filter', 'value'),
    Input('brand-filter', 'value'),
)
def update_graph(start_date, end_date, selected_sites, selected_brands):
    if selected_sites is None:
        selected_sites = []
    if selected_brands is None:
        selected_brands = []

    df = HarvestData.dropna(subset=['Site', 'Brand', 'Actual_Harvest_Date'])

    filtered = df[
        (df['Trim_Done'] == True) &
        (df['Actual_Harvest_Date'] >= pd.to_datetime(start_date)) &
        (df['Actual_Harvest_Date'] <= pd.to_datetime(end_date))
    ]

    if selected_sites:
        filtered = filtered[filtered['Site'].isin(selected_sites)]
    if selected_brands:
        filtered = filtered[filtered['Brand'].isin(selected_brands)]

    grouped = filtered.groupby(['Strain_Name', 'Brand']).agg(
        LTB_Yield_Mean=('LTB Yield', 'mean'),
        LTB_per_Plant_Mean=('LTB/Plant', 'mean'),
        LTB_Yield_SE=('LTB Yield', lambda x: x.std() / np.sqrt(len(x))),
        Count=('LTB Yield', 'count'),
        QC_Fails=('QC Fails', 'sum')
    ).reset_index()

    fig = go.Figure()

    for brand in grouped['Brand'].unique():
        brand_data = grouped[grouped['Brand'] == brand]

        fig.add_trace(go.Scatter(
            x=brand_data['LTB_per_Plant_Mean'],
            y=brand_data['LTB_Yield_Mean'],
            error_y=dict(type='data', array=brand_data['LTB_Yield_SE']),
            mode='markers+text',
            text=brand_data['Strain_Name'],
            textposition='top center',
            name=str(brand),
            marker=dict(size=12),
            customdata=np.stack([brand_data['Count'], brand_data['QC_Fails']], axis=-1),
            hovertemplate=(
                "<b>Strain:</b> %{text}<br>" +
                "<b>Brand:</b> " + brand + "<br>" +
                "<b>Avg LTB/Plant:</b> %{x:.2f} g<br>" +
                "<b>Avg LTB Yield:</b> %{y:.2f} g/ft²<br>" +
                "<b>Observations:</b> %{customdata[0]}<br>" +
                "<b>Total QC Fails:</b> %{customdata[1]} lbs"
            )
        ))

    fig.update_layout(
        title=dict(
            text="",
            x=0.5,
            xanchor='center',
            font=dict(family="Futura", size=20)
        ),
        font=dict(family="Futura", size=12),
        xaxis=dict(
            title=dict(
                text="Average LTB per Plant (g)",
                font=dict(size=14)
            ),
            tickfont=dict(size=11),
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title=dict(
                text="Average LTB Yield (g/ft²)",
                font=dict(size=14)
            ),
            tickfont=dict(size=11),
            gridcolor='lightgray'
        ),
        hovermode='closest',
        margin=dict(l=40, r=40, t=10, b=40),
        autosize=True
    )

    return fig

if __name__ == '__main__':
    app.run()
"""