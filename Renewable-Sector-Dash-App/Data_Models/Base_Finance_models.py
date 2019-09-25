# Importing webscraping packages:
import requests
import bs4
# Importing data management packages:
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf
import datetime
from datetime import timedelta
import time
import numpy as np
# Importing data vizualization packages:
import matplotlib.pyplot as plt
import seaborn as sns

sns.set() # Setting all charts to seaborn style
yf.pdr_override() # overiding pdr with yfinance packages


class Security(object):
    '''
    The Security object contains descriptive variables for each Security instance as well
    as the methods to retrive said variables from the web.

    Data is collected via the pandas_datareader package augmented with the yfinance
    package to allow data to be pulled directly from yahoo finance.

    If the Security is an Exchange Traded fund then holdings data is pulled from
    etfdb.com

    Parameters
    ----------

    ticker : str
        The string variable representing the ticker symbol for the Security. This
        string is the argument that is passed to all the data aggregation methods
    '''
    def __init__(self, ticker):

        # Declaring instance variables:
        self.ticker = ticker
        self.historical_prices = self.Price()
        self.price = round(self.historical_prices.iloc[-1]['Adj Close'], 2)

        # creating instance variable to store yfinance object:
        self.yFinance_object = yf.Ticker(self.ticker)

        # Storing specific instance variables forom yfinance object:
        self.dividend_history = self.yFinance_object.dividends
        self.title = self.yFinance_object.info['shortName']

        # Storing the historical returns dataframe:
        self.returns = self.returns()
        self.avg_return = self.returns.mean()
        self.std_return = self.returns.std()
        # Sharpe Ratio: 0.023 HISA savings account interest for risk free return
        self.sharpe_ratio = (self.avg_return - 0.023) / self.std_return

    def __repr__(self):
        return self.ticker

    def Price(self):
        '''Getting the historical price data of the ticker symbol
            from pandas_datareader

        Returns
        -------
        price : pandas dataframe
            The dataframe containing the historical price of the security from
            yahoo finance
        '''

        # Start/end date for pandas_datareader:
        start = datetime.datetime(1970, 1, 1) # Arbitrary start date
        end = datetime.datetime.today()

        # dataframe containing pricing data:
        price = pdr.get_data_yahoo(self.ticker, start, end)
        return price

    def returns(self):
        '''Method that takes the historical Adj Close price and converts it into
            a percent return on investment

        Returns
        -------
        Returns_df : pandas dataframe
            Dataframe containing the historical percent ROI for the ticker
        '''
        Returns_df = pd.DataFrame()

        # Creating column:
        Returns_df[self.ticker] = (self.historical_prices['Adj Close']).apply(
        lambda x : ((x - self.historical_prices['Adj Close'][0])) / self.historical_prices['Adj Close'][0])

        return Returns_df

class ETF(Security):
    '''
    ETF object represents an Exchange Traded Fund financial instrument. It is
    constructed as its parent class Security() and contains other fundemental
    data specific to the ETF financial instrument such as holdings data.

    Parameters
    ----------
    ticker : str
        The ticker string that is used to both initalize the parent class and to
        search for fundemental ETF data.
    '''

    def __init__(self, ticker):

        # Inheret parent __init__:
        super().__init__(ticker)

        # ETF holdings instance variables:
        self.holdings = self.build_holdings_df()

        # Constructing list of Security() objects from ETF ticker holdings:
        self.holdings_list = self.build_holdings_objects()

        # dataframe comparing the ROI of the ETF to its top 10 holdings:
        self.holdings_ROI = self.build_holdings_comparions()

    def build_holdings_df(self):
        '''Method uses pd.read_html to extract top 10 holdings table from
            Yahoo Finance website based on self.ticker

        Returns
        -------
        pd.read_html(url)[0] : pandas dataframe
            Dataframe containing the top 10 holdings of the ETF with: Name,
            Ticker symbol and % Allocation
        '''

        # Building the Yahoo Finance holdings tab url:
        url = "https://ca.finance.yahoo.com/quote/{self.ticker}/holdings?p={self.ticker}".format(self=self)

        # Creating a dataframe from the webpage:
        return pd.read_html(url)[0] # converting list of 1 df to dataframe

    def build_holdings_objects(self):
        '''Method that extracts a list of ticker symbols from the self.holdings
            dataframe and attempts to initalize each ticker as a Security() object

        Returns
        -------
        holdings_list : lst
            The list containing all the Security() objects from the self.holdings
            dataframe
        '''

        # Creating the main list:
        holdings_list = []

        # Calling the dataframe:
        df = self.holdings

        # iterating through the dataframe and appending to the list:
        for ticker in df['Symbol']:

            try: # attempting to initalize Security() object
                ticker = Security(ticker)

            except: # TODO: Write the ticker_cleaning method for exceptions:
                ticker = 'NaN'

            # Creating the list of Security() objects
            holdings_list.append(ticker)

        return holdings_list

    def build_holdings_comparions(self):
        '''Method extracts the performance of the top 10 holdings of an ETF and
            constructs a dataframe comparing the YTD performance of of each holding
            to the overall performance of the ETF

        Returns
        -------
        holdings_YTD_df : pandas dataframe
            The dataframe containing the YTD performance of each top 10 security
            holdings of the ETF and the YTD performance of the ETF
        '''

        # Creating dataframe:
        holdings_YTD_df = pd.DataFrame()

        holdings_YTD_df = self.returns

        # Adding columns to dataframe:
        holdings_YTD_df[self.ticker] = self.returns

        # Creating a df column for every holding in the Etf holdings list:
        for Security in self.holdings_list:

            try:
                holdings_YTD_df[Security.ticker] = Security.returns
            except:
                holdings_YTD_df['NaN'] = Security # TODO: Deal with erroring out

        # Really sketchy error handeling lol:
        holdings_YTD_df.drop(['NaN'], axis=1)

        # Converting 'NaN' values to 0 for visual clarity:
        holdings_YTD_df.fillna(0)

        return holdings_YTD_df

class REIT_Sector(object):
    '''
    REIT_Sector object represens an economic sector of the REIT market. It
    contains the categorization of the sector as well as all the REIT's within
    the sector stored as REIT() objects.

    Parameters
    ----------
    Sector : str
        The economic sector being described. eg: 'Logging'

    *REITs
        The ticker symbol for each REIT held within the sector. These variables
        will be initalized as a REIT() object

    '''
    def __init__(self, Sector, *REITs):

        self.Sector = Sector

        # Loop to iterate over *REIT, Initializing the REIT() objects and appending
        # them to a list:
        Sector_holdings = []
        for ticker in REITs:
            ticker = REIT(ticker)
            Sector_holdings.append(ticker)

        # Declaing the obejct variable as a list:
        self.Sector_holdings = Sector_holdings

        # Declaring comparitive dataframe variables:
        self.YTD_performance_comparrison = self.build_YTD_performance()
        self.historical_dividend_comparrison = self.build_historical_dividend_performance()

    def build_YTD_performance(self):
        '''Method constructs a dataframe comparing the YTD performance of all REIT
            objects in the list self.Sector_holdings

        Returns
        YTD_returns : pandas dataframe
            The dataframe containing the performances of each REIT
        '''
        # Creating dataframe:
        YTD_returns = pd.DataFrame()

        # loop iterating over Sector_holdings list building each df column:
        for REIT in self.Sector_holdings:

            # Series for Adj close prices of current year:
            Adj_Close = REIT.historical_prices['Adj Close']\
             [REIT.historical_prices.index.year == datetime.datetime.now().year]

            # Calculating Returns based on the current year's first price:
            Adj_Close_Returns = Adj_Close.apply(lambda x: ((x - Adj_Close[0])/Adj_Close[0]))

            # Appending series to YTD_returns:
            YTD_returns[REIT.ticker] = Adj_Close_Returns


        return YTD_returns

    def plot_YTD(self):
        '''The Graphing method that graphs the YTD performance of every REIT within
             the sector using the YTD_comparrison dataframe self.YTD_performance_comparrison

        self.build_YTD_performance datframe follows pandas formatting convention
        so the .plot() method can be used for simplicity

        '''
        self.YTD_performance_comparrison.plot()

        # Formatting graph:
        plt.legend(loc=2)
        plt.ylabel('% Yield')
        plt.xlabel('Date')
        plt.title('{} REIT Sector'.format(self.Sector))

        plt.show()

    def plot_div(self):
        '''Plots the historical dividend performance of each REIT in a subplot

        Raises
        ------
        FutureWarning: Using an implicitly registered datetime converter for a
         matplotlib plotting method. The converter was registered by pandas on
         import. Future versions of pandas will require you to explicitly
         register matplotlib converters.
        '''
        # Initializing the figue and number of axis:
        fig, axs = plt.subplots(len(self.Sector_holdings))

        # Loop to iterate over the *REITs plotting the subplots:
        counter = 0
        for REIT in self.Sector_holdings:

            # c= np.random.rand(3,) generates random unique color for each plt:
            axs[counter].plot(REIT.dividend_history, c=np.random.rand(3,))
            axs[counter].set_title(REIT.title)

            counter = counter + 1


        for ax in axs.flat:
            ax.label_outer() # hiding axis tick labels for top plots

        plt.show()

class Quarterly_Report(object):
    """
    The Quarterly_Report object is the parent report object that contains
    several key variables that will be used by several child report type
    objects.

    The instance variables that this object stores are as follows:

    Returns
    ------
    self.pref_quarter : String
        This string is constructed from a categorical variable as well as a
        datetime object to describe the most recent pervious financial quarter

    self.quarter_period : pandas date_range()
        This date_range object stores the range of dates present within the
        specified quarter

    self.avg_market_returns : pandas dataframe
        The dataframe containing the ROI of the S&P500 during the quarterly
        period
    """

    def __init__(self):

        # returns dict generated by object method:
        self.prev_quarter_dict = self.get_previous_quarter()

        # Extracts values from method generated dict above:
        self.quarter = self.prev_quarter_dict['Quarter']
        self.quarter_range = self.prev_quarter_dict['daterange']


        # S&P 500 quarterly returns and volatility metrics:
        self.avg_market_returns = self.get_avg_market_returns() # inital dataset
        self.mean_market_returns = self.avg_market_returns.mean() # mean
        self.market_retrun_stdv = self.avg_market_returns.std() # standard deviation

        # Sharpe Ratio:  0.023 HISA savings account interest for risk free return
        self.market_sharpe = (self.mean_market_returns - 0.023) / self.market_retrun_stdv


    def get_previous_quarter(self):
        """get_previous_quarter method uses datetime objects and the timedelta
            function to build a string that indicates what the most recent
            previous quarter was

        Returns
        -------
        dict : dict
            A dictionary containing both the most recent quarter and the date
            range of said period
        """

        def quarter_range(year):
            """Internal method that generates a dictionary containing the date
                ranges for all fiscal quarters of a given year:

            Receives
            --------
            year: str
                The year that will be converted to a datetime object and used to
                determine the date ranges of the fisical quarters

            Returns
            -------
            Quarters : dict
                Dictionary containing the date ranges of all four fiscal quarters
                for the given year
            """


            # Building Q dates:
            Q1_start = '1/01/{}'.format(year)
            Q1_end = '30/03/{}'.format(year)

            Q2_start = '1/04/{}'.format(year)
            Q2_end = '30/06/{}'.format(year)

            Q3_start = '1/07/{}'.format(year)
            Q3_end = '30/09/{}'.format(year)

            Q4_start = '1/10/{}'.format(year)
            Q4_end = '30/12/{}'.format(year)

            # Creating date ranges:
            Q1 = pd.date_range(start= Q1_start, end=Q1_end)
            Q2 = pd.date_range(start= Q2_start, end=Q2_end)
            Q3 = pd.date_range(start= Q3_start, end=Q3_end)
            Q4 = pd.date_range(start= Q4_start, end= Q4_end)

            # constructing a dict storing all these variables:
            Quarters = {'Q1': Q1,
                        'Q2': Q2,
                        'Q3': Q3,
                        'Q4': Q4
                        }

            return Quarters

        # Declaring date variables:
        current_date = datetime.date.today()
        current_year = datetime.datetime.now().year
        prev_year = (datetime.datetime.now() - timedelta(days=365)).year


        # Creating an instance of quarter_range for current_year:
        current_quarters = quarter_range(current_year)
        # Creating an instance of quarter_range for previous year:
        prev_quarters = quarter_range(prev_year)


        # Creating Conditional that returns the data dictionary:
        # NOTE: if current date is in Q1, then most recent quarter is Q4 of last
        # year: -> The selected quarter is the previous quarter.
        if current_date in current_quarters['Q1']:
            dict = {'Quarter': '4Q' + str(prev_year),
                    'daterange': prev_quarters['Q4']}

            return dict

        elif current_date in current_quarters["Q2"]:
            dict = {'Quarter': '1Q' + str(current_year),
                    'daterange': current_quarters['Q1']}

            return dict

        elif current_date in current_quarters['Q3']:
            dict = {'Quarter': '2Q' + str(current_year),
                    'daterange': current_quarters['Q2']}

            return dict

        else:
            dict = {'Quarter': '3Q' + str(current_year),
                    'daterange': current_quarters['Q3']}
            return dict

    def get_avg_market_returns(self):
        """Method takes the complete historical prices of the S&P500 and extracts
            a dataframe of pricing data from the last quarterly period. It then
            transforms this historical adjusted closed quarterly_price data from
            absoloute price to ROI.

        Returns
        -------
        quarterly_return : pandas dataframe
            The dataframe containing the ROI of the S&P500 in the previously
            declared fisical quarter period
        """

        # ^GSPC (S&P 500) object:
        SP_500 = Security('^GSPC')

        # Extracting the Adj Close prices from the previous fisical quarter:

        # Declaring variables for clarity:
        SP_500_price = SP_500.historical_prices['Adj Close']

        # Using isin() function to select pd.daterange from dataframe:
        quarterly_prices = SP_500_price[SP_500_price.index.isin(self.quarter_range)]

        # Converting price dataframe to percent ROI:
        quarterly_return = quarterly_prices.apply(lambda x: (x - quarterly_prices[0])
         / quarterly_prices[0]).rename(columns = {'Adj Close': 'ROI'}, inplace=True)


        return quarterly_return

class Sector_Quarterly_Report(Quarterly_Report):
    """
    The Sector_Quarterly_Report() object inherits from the parent object
    Quarterly_Report() and is used to represent the quarterly comparion
    between the "market" (S&P500) and an economic sector.

    This is done by comparing performance metrics between the market and a ETF
    that is initally input into the class. This ETF is used to roughtly
    represent the economic sector deffined by an instance string variable.


    Parameters
    ----------
    sector : str
        A string used for purely definitional purposes. Eg 'Wind' or 'Solar'

    sector_ETF : str
        The string of the ETF ticker symbol meant to represent the specific
        economic sector. Sector_ETF string wil be used to initalize a Security()
        object
    """

    def __init__(self, sector, sector_ETF):

        # Initializing parent class:
        super().__init__()
        self.sector = sector # Creating instance sector deffinition


        # Initalizing Sector specific ETF:
        self.Sector_ETF = ETF(sector_ETF)

        # The dataframe containing each Security's performance metrics:
        self.performance = self.build_performance_dataframe()


    def performance_metrics(self, Security_object):
        """This nested method returns a dictionary with all the necessary
        quarterly metrics for a perviously inialized Security object within the
        current quarterly period. It is only called within other methods within
        the objet and is not initalized with the object.

        Parameters
        -----------
        Security_object : Security() object
            The security object that provids the fundemental pricing data that
            will be used to calculate perfromace metrics.


        Returns
        -------
        metrics : dictionary
            The dataframe that contains all of the quarterly performace
            metrics for the given quarter
        """

        # Creating instances of Security() variables within the quarterly period:
        price = Security_object.historical_prices['Adj Close']

        price = price[price.index.isin(self.quarter_range)]

        Security_returns = price.apply(lambda x: (x-price[0])
        /price[0]).rename({'Adj Close' : 'ROI'})


        # Single quarterly performace metrics:
        Securty_final_return = Security_returns[-1]
        Security_avg_return = Security_returns.mean()
        Security_Sharpe = (Security_avg_return - 0.023 ) / Security_returns.std()

        # TODO: Add volatility calculation
        # TODO: Add Alpha calculation

        # Crating the dictionary that will contain all the return variables:
        metrics = {

                    'returns_df': Security_returns,
                    'final_return': Securty_final_return,
                    'avg_return': Security_avg_return,
                    'sharpe' : Security_Sharpe

                    }

        return metrics

    def build_performance_dataframe(self):
        """Method builds a pandas dataframe containing all the relevant
            quarterly performance metrics for the Sector specific ETF and its
            holdings.

            The performance metrics used to compare Security() objects:
            - Sharpe Ratio
            - Average Return
            - Investment Return at the end of quarter
            - Volatility during quarter
            - Alpha compared to S&P500

            Returns
            --------
            performance_df : pandas dataframe
                A dataframe that contains all of the quarterly performance
                metrics for the sector specific security and its holdings
        """
        # Creating empty dataframe:
        performance_df = pd.DataFrame(columns = ['Security', 'Ticker', 'Sharpe_ratio',
         'avg_return', 'Fin_return'])

        # Initalizing ETF data
        ETF = self.performance_metrics(self.Sector_ETF)

        # Appending market data to dataframe:
        performance_df = performance_df.append({'Security': 'S&P 500 Index',
                                                'Ticker' : '^GSPC',
                                                'Sharpe_ratio': self.market_sharpe,
                                                'avg_return': self.mean_market_returns,
                                                'Fin_return': self.avg_market_returns[-1]
                                                # TODO: Add Volatility to dataframe
                                                # TODO: Add Alpha to dataframe
                                                }, ignore_index=True)

        # Appending ETF data to dataframe:
        performance_df = performance_df.append({'Security': self.Sector_ETF.title,
                                                'Ticker': self.Sector_ETF.ticker,
                                                'Sharpe_ratio': ETF['sharpe'],
                                                'avg_return': ETF['avg_return'],
                                                'Fin_return': ETF['final_return']
                                                # TODO:  Add Volatility to dataframe
                                                # TODO: Add Alpha to dataframe
                                                }, ignore_index=True)


        # Creating an iterative loop that adds performace metrics of the ETF
        # holdings to the perfromace dataframe:
        for Security in self.Sector_ETF.holdings_list:

            # Extracting performance metrics from each Security instance:

            try: # If Security object cannot be initalized, returns 'NaN'
                metrics = self.performance_metrics(Security)
            except: # dictionary that retruns 'NaN' for all failed values
                metrics = 'NaN'

            try:
                # Appending said performance metrics to the performance_df:
                performance_df = performance_df.append(
                {'Security': Security.title,
                'Ticker' : Security.ticker,
                'Sharpe_ratio' : metrics['sharpe'],
                'avg_return' : metrics['avg_return'],
                'Fin_return' : metrics['final_return']
                # TODO: Add Volatility to dataframe
                # TODO: Add Alpha to dataframe
                 }, ignore_index=True)

            # populating df with null values if Security object fails to inialize
            except:
                performance_df = performance_df.append(
                    {'Security': 'NaN',
                    'Ticker' : 'NaN',
                    'Sharpe_ratio' : 'NaN',
                    'avg_return' : 'NaN',
                    'Fin_return' : 'NaN'
                    # TODO: Add Volatility to dataframe
                    # TODO: Add Alpha to dataframe
                     }, ignore_index=True)


        # Setting index:
        performance_df.set_index('Ticker', inplace=True)
        # Removing values that failed to initalize:
        performance_df = performance_df[performance_df.index != 'NaN']

        return performance_df
