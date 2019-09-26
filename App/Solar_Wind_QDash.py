# Importing finance models:
from Base_Finance_models import Sector_Quarterly_Report
# Importing dash packages:
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

# Initalizing The Quarterly_Report objects for the desired sectors:
TAN = Sector_Quarterly_Report('Solar Energy', 'TAN')

# TODO: Add CSS styling
# TODO: Complete the rest of the Dash application based on the written outline.

app = dash.Dash()

# Formatting dashboard:
app.layout = html.Div([

    # tabs:
    dcc.Tabs(id='tabs', children=[
        # Solar industry, TAN ETF:
        dcc.Tab(label='TAN', children=[
        # Building Solar Dashboard:

            # First Div tag containing ETF summmary performace:
            html.Div(children=[
                html.H1(
                    children= TAN.Sector_ETF.title + ' (' + TAN.Sector_ETF.ticker + ')',
                    style={
                        'textAlign':'left',
                    }
                ),
                # Necessary ETF information:
                html.Div(children='Current Price: $' + str(TAN.performance.Price[1])),
                html.Div(children='Quarter Return: ' + str(TAN.performance.Fin_return[1])),
                html.Div(children='Sharpe Ratio: ' + str(TAN.performance.Sharpe_ratio[1]))
            ]),

            # Second Div tag containing the market summary performace:
            html.Div(children=[
                html.H1(
                    children= TAN.performance.Security[0] + ' (' + TAN.performance.index[0] + ')',
                ),
                # Necessary Market information:
                html.Div(children='Current Price: $' + str(TAN.performance.Price[0])),
                html.Div(children='Quarter Return: ' + str(TAN.performance.Fin_return[0])),
                html.Div(children='Sharpe Ratio: ' + str(TAN.performance.Sharpe_ratio[0]))
            ]),

        ])
    ])
])

# Debug:
print(TAN.performance)

if __name__ == '__main__':
    app.run_server(debug=True)
