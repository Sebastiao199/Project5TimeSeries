# Imports

# from jupyter_dash import JupyterDash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api
import datetime
from datetime import datetime

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PIL import Image
import base64

from dtw import *
from scipy import stats

import flag

####################################################################################################

### Interactive plot for Lisbon - Rent m2

df_lisbon = pd.read_csv('https://raw.githubusercontent.com/Sebastiao199/Project5TimeSeries/main/idealista_df_rent.csv')

df_lisbon['Date']= pd.to_datetime(df_lisbon['Date']).dt.strftime('%Y-%m-%d')
df_lisbon.sort_values(by='Date', inplace = True)

trace2 = go.Scatter(x=df_lisbon['Date'],
                    y=df_lisbon['Rent_Squared_Meter_Price'],
                    mode='lines',
                    line=dict(width=1.5))

frames2 = [dict(data= [dict(type='scatter',
                           x=df_lisbon['Date'][:k+1],
                           y=df_lisbon['Rent_Squared_Meter_Price'][:k+1]),
                     ],
               traces= [0, 1, 2, 3],  
              )for k  in  range(1, len(df_lisbon)-1)]
layout2 = go.Layout(width=900,
                   height=600,
                   showlegend=False,
                   hovermode='x unified',
                   title=go.layout.Title(text="Evolution of the price of the rent per m2 in Lisbon from January 2019 to December 2022"),
                   updatemenus=[
                        dict(
                            type='buttons', showactive=False,
                            y=1.05,
                            x=1.15,
                            xanchor='right',
                            yanchor='top',
                            pad=dict(t=0, r=10),
                            buttons=[dict(label='Play',
                            method='animate',
                            args=[None, 
                                  dict(frame=dict(duration=90, 
                                                  redraw=False),
                                                  transition=dict(duration=0),
                                                  fromcurrent=True,
                                                  mode='immediate')]
                            )]
                        ),
                    ]              
                  )
layout2.update(xaxis =dict(range=['2019-01-01','2022-12-15'], autorange=False),
              yaxis =dict(range=[12, 19], autorange=False));
fig_lisbon = go.Figure(data=[trace2], frames=frames2, layout=layout2)


### Interactive plot for San Francisco - Rent index
 
df_sf = pd.read_csv('https://raw.githubusercontent.com/Sebastiao199/Project5TimeSeries/main/rent_index_sf.csv')
df_sf.rename(columns={'Unnamed: 0': 'DATE', 'Rent Index' : 'Rent_Index'},
          inplace=True, errors='raise')
df_sf['DATE']= pd.to_datetime(df_sf['DATE']).dt.strftime('%Y-%m-%d')
df_sf.sort_values(by='DATE', inplace = True)
df_sf['Rent_Index'] = df_sf['Rent_Index'].interpolate(method='nearest')

trace1 = go.Scatter(x=df_sf['DATE'],
                    y=df_sf['Rent_Index'],
                    mode='lines',
                    line=dict(width=1.5))

frames = [dict(data= [dict(type='scatter',
                           x=df_sf['DATE'][:k+1],
                           y=df_sf['Rent_Index'][:k+1]),
                     ],
               traces= [0, 1, 2, 3],  
              )for k  in  range(1, len(df_sf)-1)]
layout = go.Layout(width=900,
                   height=600,
                   showlegend=False,
                   hovermode='x unified',
                   title=go.layout.Title(text="Evolution of the Rent index in San Francisco from March 2015 to November 2022"),
                   updatemenus=[
                        dict(
                            type='buttons', showactive=False,
                            y=1.05,
                            x=1.15,
                            xanchor='right',
                            yanchor='top',
                            pad=dict(t=0, r=10),
                            buttons=[dict(label='Play',
                            method='animate',
                            args=[None, 
                                  dict(frame=dict(duration=90, 
                                                  redraw=False),
                                                  transition=dict(duration=0),
                                                  fromcurrent=True,
                                                  mode='immediate')]
                            )]
                        ),
                    ]              
                  )
layout.update(xaxis =dict(range=['2015-03-31','2022-12-30'], autorange=False),
              yaxis =dict(range=[2411.441275,3200], autorange=False));

fig_sf = go.Figure(data=[trace1], frames=frames, layout=layout)




# Dynamic Time Warping 


rent_sf_without_date = df_sf['Rent_Index']
rent_lx_without_date = df_lisbon['Rent_Squared_Meter_Price']

# arr_sf = rent_sf_without_date.to_numpy()
# arr_lx = rent_lx_without_date.to_numpy()

list_index_lx = list(range(0,46))

def check_correlations(lx_index):
  x=48

  # replaced rent_lx with rent_lx_without_date
  intermediate_dictionary = {'Lisboa':df_lisbon['Rent_Squared_Meter_Price'], 'SanFrancisco':rent_sf_without_date[lx_index:x+lx_index]}

  pandas_dataframe = pd.DataFrame(intermediate_dictionary)

  return pandas_dataframe.corr().iloc[0,1]

result = list(map(check_correlations, list_index_lx))

corr_sf_lx = pd.DataFrame(result,columns=['Correlation'])

max_corr_value = corr_sf_lx['Correlation'].max()
max_corr = corr_sf_lx[corr_sf_lx['Correlation'] == max_corr_value]

max_corr_df = pd.merge(df_sf['DATE'], rent_sf_without_date[max_corr.index[0]:48+max_corr.index[0]], left_index=True, right_index=True)
max_corr_df.sort_values(by='DATE', inplace = True)

fig=make_subplots(
        specs=[[{"secondary_y": True}]])
# print(fig.layout)    

fig.update_layout(xaxis2= {'anchor': 'y', 'overlaying': 'x', 'side': 'top'},
                   yaxis_domain=[0, 0.94]);

fig.add_trace(
    go.Scatter(x=max_corr_df['DATE'], 
           y=max_corr_df['Rent_Index'],
           mode='lines',
           name="San Francisco",
          ), secondary_y=False)

fig.add_trace(
    go.Scatter(x=df_lisbon['Date'],
               y=df_lisbon['Rent_Squared_Meter_Price'],
               mode='lines',
               name="Lisbon",
               line_color="#ee0000"), secondary_y=True)

# layout_dtw = go.Layout(width=900,
#                    height=600,
#                    showlegend=False,
#                    hovermode='x unified',
#                    title=go.layout.Title(text="Is there a Correlation between San Francisco and Lisbon?"),        
#                   )

fig.data[1].update(xaxis='x2')
fig_dtw = fig.update_layout(width=900, height=600, title="Is there a Correlation between San Francisco and Lisbon?")


# Forecasting 

# Lisbon 

df_lisbon.set_index('Date', inplace=True)

# fit model
arima_model = ARIMA(df_lisbon['Rent_Squared_Meter_Price'], order=(5,1,0))
arima_model_fit = arima_model.fit()

# Split into train and test sets
X = df_lisbon['Rent_Squared_Meter_Price'].values

size = int(len(X) * 0.75)
train, test = X[0:size], X[size:len(X)]

history = [x for x in train]

predictions = []

# Walk-forward validation
for t in range(len(test)):
	model = ARIMA(history, order=(1,0,2))
	model_fit = model.fit()
	output = model_fit.forecast()

	pred_y = output[0]
	predictions.append(pred_y)

	obs_y = test[t]

	history.append(obs_y)

	# print('Actual=%f, Predicted=%f' % (obs_y, pred_y))

df_test_predictions_lisbon = pd.DataFrame({'test':test, 'prediction':predictions})
df_test_predictions_lisbon.index=df_lisbon.index[-12:]

# FINAL PLOT LISBON

fig=make_subplots(
        specs=[[{"secondary_y": True}]])
# print(fig.layout)
fig.update_layout(xaxis2= {'anchor': 'y', 'overlaying': 'x', 'side': 'top'},
                   yaxis_domain=[0, 0.94]);

# LISBON

# Predicted
fig.add_trace(
    go.Scatter(x=df_lisbon.index[-12:],
               y=df_test_predictions_lisbon['prediction'],
               mode='lines',
               name="Lisbon predicted",
               line_color="orange"), secondary_y=True)


fig.add_trace(
    go.Scatter(x=df_lisbon.index,
               y=df_lisbon['Rent_Squared_Meter_Price'],
               mode='lines',
               name="Lisbon actual values",
               line_color="blue"), secondary_y=True)

fig.update_layout(width=900, height=600, 
                  title= 'Prediction of the evolution of the price per squared meter in Lisbon',
                  xaxis_title="Date",
                  yaxis_title="Price")

fig_forecast_lisbon = fig.update_yaxes(range=[0,20])


# San Francisco

df_sf.set_index('DATE', inplace=True)

arima_model = ARIMA(df_sf['Rent_Index'], order=(5,1,0))
arima_model_fit = arima_model.fit()

# Split into train and test sets
X = df_sf['Rent_Index'].values

size = int(len(X) * 0.75)
train, test_sf = X[0:size], X[size:len(X)]

history = [x for x in train]

predictions_sf = []

# Walk-forward validation
for t in range(len(test_sf)):
	model = ARIMA(history, order=(3,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()

	pred_y = output[0]
	predictions_sf.append(pred_y)

	obs_y = test_sf[t]

	history.append(obs_y)

	# print('Actual=%f, Predicted=%f' % (obs_y, pred_y))

df_test_predictions_sf = pd.DataFrame({'test':test_sf, 'prediction':predictions_sf})
df_test_predictions_sf.index=df_sf.index[-24:]

fig=make_subplots(
        specs=[[{"secondary_y": True}]])
# print(fig.layout)
fig.update_layout(xaxis2= {'anchor': 'y', 'overlaying': 'x', 'side': 'top'},
                   yaxis_domain=[0, 0.94]);

# SF

# Predicted
fig.add_trace(
    go.Scatter(x=df_sf.index[-24:],
               y=df_test_predictions_sf['prediction'],
               mode='lines',
               name="San Francisco predicted",
               line_color="orange"), secondary_y=True)

fig.add_trace(
    go.Scatter(x=df_sf.index,
               y=df_sf['Rent_Index'],
               mode='lines',
               name="San Francisco actual values",
               line_color="blue"), secondary_y=True)

fig_forecast_sf = fig.update_layout(width=900, height=600,
                  title= 'Prediction of the evolution of the rent index in San Francisco',
                  xaxis_title="Date",) 


# Images Emojis 

pt_flag = flag.flag("PT")
usa_flag = flag.flag("US")

# Start of the App 


# Add the row + column
app = Dash(external_stylesheets=[dbc.themes.LUX])


# STYLE
H1={
    "backgroundColor": "blue",
    "padding": 16,
    "marginTop": 32,
    "textAlign": "center",
    "fontSize": 32,
}

#images
 #image_filename = 'picto_intro.png' # replace with your own image
 #encoded_image = base64.b64encode(open(image_filename, 'rb').read())
image_path = 'picto_intro.png'




# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

    # First dataset with Lisbon
# df = pd.read_csv('https://raw.githubusercontent.com/Sebastiao199/Project5TimeSeries/main/idealista_df_rent.csv')
# df['Date']= pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
# df.sort_values(by='Date', inplace = True)

    # Second Dataset with SF:
# df_lisbon = pd.read_csv('https://raw.githubusercontent.com/Sebastiao199/Project5TimeSeries/main/idealista_df_rent.csv')
# df_lisbon['Date']= pd.to_datetime(df_lisbon['Date']).dt.strftime('%Y-%m-%d')
# df_lisbon.sort_values(by='Date', inplace = True)

# style = {'margin':20px'},
    # LAYOUT
app.layout = dbc.Container(
    [
        dcc.Store(id="store"),
        html.H1("Real estate & Digital Nomads 🏡"),
        html.H2("Is Lisbon the New San Francisco?"),
        html.Hr(),
        html.Img(src=app.get_asset_url('picto_intro.png'), height='120', width='120'),
        #html.Img(src=app.get_asset_url('/content/drive/MyDrive/WILD CODE SCHOOL/picto_intro.png')),
        #html_Img(src='/content/drive/MyDrive/WILD CODE SCHOOL/picto_intro.png'),
        #html.Img(src='https://github.com/Sebastiao199/Project5TimeSeries/blob/main/picto_intro.png',base64,{}.format(encoded_image)),
        #html.Img(src='https://github.com/Sebastiao199/Project5TimeSeries/blob/main/picto_intro.png;base64,{}'.format(encoded_image)),
        #html.Img(src=image_path),

# Carrousel try       
#         dbc.Carousel(
#     items=[
#         {"key": "1", "src": "https://github.com/Sebastiao199/Project5TimeSeries/blob/main/picto_intro.png"},
#         {"key": "2", "src": "picto_intro.png"},
#         {"key": "3", "src": "/static/images/slide3.svg"},
#     ],
#     controls=True,
#     indicators=False,
# ),
# Tabs
        dbc.Tabs(
            [
                dbc.Tab(label=f"San Francisco {usa_flag}", tab_id="scatter"),
                dbc.Tab(label=f"Lisbon {pt_flag}", tab_id="histogram"),
                dbc.Tab(label="Forecast 🔮", tab_id="forecast"),
                dbc.Tab(label="San Francisco vs Lisbon 🥊", tab_id="prediction"),
            ],
            id="tabs",
            active_tab="scatter",
        ),
        html.Div(id="tab-content", className="p-4"),
     
 # Contact button    
        dbc.Button(
            "Contact Rita for more information",
            color="primary",
            id="button",
            className="mb-3",
        ),
     
    ]
)

@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("store", "data")],
)
# Function to display the graphs
def render_tab_content(active_tab, data):
    if active_tab and data is not None:
        if active_tab == "scatter":
            return dcc.Graph(figure= go.Figure(data=[trace1], frames=frames, layout=layout))
        elif active_tab == "histogram":
            return dcc.Graph(figure= go.Figure(data=[trace2], frames=frames2, layout=layout2))
        elif active_tab == "forecast":
            return dcc.Graph(figure= fig_forecast_lisbon), dcc.Graph(figure= fig_forecast_sf)
        elif active_tab == "prediction":
            return dcc.Graph(figure= fig_dtw)
    return "No tab selected"

@app.callback(
    Output("store", "data"), 
    Input("button", "n_clicks"),
)
def generate_graphs(n):
    if not n:
        # generate empty graphs when app loads
        return {k: go.Figure(data=[]) for k in ["scatter"]}

# save figures in a dictionary for sending to the dcc.Store
# return {"scatter": scatter}
   
# HOW THEY CHANGE



# finish
if __name__ == '__main__':
    app.run_server(debug=True)