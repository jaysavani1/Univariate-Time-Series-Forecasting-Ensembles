# -*- coding: utf-8 -*-
"""autotsforecastML.py

Structure:

class a:
    methode1():
        submethode():
            # some code
            # some code 
        # some code
        
    methode2():
        submethode():
            # some code
            # some code
        # some code
        .
        .
        .
    methode3():
        submethode():
            # some code
            # some code
        # some code
"""
# Load Modules
# system
import warnings
warnings.filterwarnings("ignore")

import logging
logging.disable(logging.CRITICAL)

import sys, numbers, math
from time import perf_counter
import datetime as dt

# for data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from functools import reduce
from typing import List, Tuple, Union
from tqdm.auto import tqdm

# forecasting helper
import pmdarima as pmd
import statsmodels.api as sm 
from scipy.stats import normaltest

# darts
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.models import (
    ARIMA,
    AutoARIMA,
    ExponentialSmoothing,
    NaiveSeasonal,
    NaiveDrift,
    Prophet,
    RegressionEnsembleModel,
    Theta
)
from darts.dataprocessing.transformers.boxcox import BoxCox
from darts.metrics import mape, mase, mae, mse, ope, r2_score, rmse, rmsle
from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis
from darts.utils.utils import ModelMode
from darts.datasets import ( 
    AirPassengersDataset,
    ElectricityDataset
)

class AutoUnivariateiTS:
    """This class helps users to implement multiple SOTA statistical time series forecasting models.
    
    Description:
    -----------

        Supported Methods:
        -----------------
            - pandasdf_to_timeriesdata(**kwargs) : Convert pandas dataframe to dtype timeries data.
            - timeriesdata_to_pandasdf(**kwargs) : Convert dtype timeries data to pandas dataframe.
            - timeriesdata_to_pdseries(**kwargs) : Convert dtype timeries data to pandas series.
            - seasonality_check(**kwargs)        : Check if there is any seasonality present in given timeseries.
            - train_test_split_data(**kwargs)    : Split data into training and validation set to perform model forecasting.
            - fit_predict(**kwargs)              : Produce accuracy metrics for the selected or all default models.
            - plot_fit_predict(**kwargs)         : Plot the predictions derived by fit_predict().
            - plot_residual_diagnostics(**kwargs): Create various visualizations based on models to study residuals.
        
        Future Methods:
        --------------
            - predict(**kwargs)    : predict the future timestamps.
            - save_model(**kwargs) : Save the model.
            - load_model(**kwargs) : Load the saved model.
    """

    def __init__(
        self,
        trace: bool = False
    ) -> None:

        self.TRACE = trace

    def pandasdf_to_timeriesdata(
        self,
        data: pd.DataFrame,
        target_column: List[str] = [],
        index_col: str = None
    ):
         """Convert pandas dataframe to dtype timeries data.

        Args:
            data (pd.DataFrame): Enter pandas dataframe.
            target_column (List[str]): Enter the target or dependent(y) column inside a list. Defaults to Empty list.

        Returns:
            darts.timeseries.TimeSeries: return timeseries data object from darts.
        """
        if not target_column:
            if isinstance(data.index, pd.DatetimeIndex):
                data = data.copy()
                tseries = TimeSeries.from_dataframe(data, value_cols=data.columns)
            
            elif index_col:
                data = data.copy().set_index(index_col)
                tseries = TimeSeries.from_dataframe(data, value_cols=data.columns)

            else:
                raise ValueError("""
                    Invalid index column. Set valid index column using 'index_col' parameter !!!
                """)
        else:
            if isinstance(data.index, pd.DatetimeIndex):
                data = data.copy()
                tseries = TimeSeries.from_dataframe(data, value_cols=target_column)
            
            elif index_col:
                data = data.copy().set_index(index_col)
                tseries = TimeSeries.from_dataframe(data, value_cols=target_column)

            else:
                raise ValueError("""
                    Invalid index column. Set valid index column using 'index_col' parameter !!!
                """)
        self._data = tseries
        return tseries

    def timeriesdata_to_pandasdf(
        self,
        data: darts.timeseries.TimeSeries,
        reset_index: bool = False
    ):
        """Convert dtype timeries data to pandas dataframe.

        Args:
            data (darts.timeseries.TimeSeries): Enter darts or xarray time series data object.
            reset_index (bool, optional): Reset index on conveted pandas dataframe. Defaults to False.

        Returns:
            pd.DataFrame: Return pandas dataframe object.
        """
        if not reset_index:
            pdf = data.pd_dataframe()
        else:
            pdf = data.pd_dataframe().reset_index()
        
        return pdf

    def timeriesdata_to_pdseries(
        self,
        data: darts.timeseries.TimeSeries,
    ):
        """Convert dtype timeries data to pandas series.

        Args:
            data (darts.timeseries.TimeSeries): Enter darts or xarray time series data object.

        Returns:
            pd.series: Return pandas series object.
        """
        pdseries = data.pd_series()
        return pdseries

    def seasonality_check(
        self,
        data: darts.timeseries.TimeSeries,
        seasonality_start: int = 2,
        seasonality_end: int = 50,
        alpha = 0.05,
        print_summary: bool = True
    ):
        """Check if there is any seasonality present in given timeseries.

        Args:
            data (darts.timeseries.TimeSeries): Enter darts timeseries data object.
            seasonality_start (int, optional): Enter starting range value to perform seasoanlity check. Defaults to 2.
            seasonality_end (int, optional): Enter ending range value to perform seasoanlity check. Defaults to 50.
            alpha (float, optional): Enter the significance confidence interval. Defaults to 0.05.
            print_summary (bool, optional): Print out the summary results. Defaults to True.
        """
        
        self.ALPHA = alpha
        for m in range(seasonality_start, seasonality_end):
            is_seasonal, MSEAS = check_seasonality(data, m=m, alpha=alpha)
            if is_seasonal:
                break
        self.is_seasonal = is_seasonal
        self.MSEAS = MSEAS
        if print_summary:
            print(f"Is provided data seasonal? :{self.is_seasonal}")
            if self.is_seasonal:
                print(f"There is seasonality of order {self.MSEAS}")

    def train_test_split_data(
        self,
        data: darts.timeseries.TimeSeries,
        spliting_at: Union[pd.Timestamp, int, float],
        split_before: bool = True,
        plot: bool = True,
        plot_size = (12, 5),
        legend_loc = 'upper right'
    ):
        """_summary_

        Args:
            data (darts.timeseries.TimeSeries): Enter darts timeseries data object.
            spliting_at (Union[pd.Timestamp, int, float]): Split the data either from perticular timestamp, or enter proportion of splitting ratio.
                - split position: if string, then interpret as Timestamp
                - if int, then interpretation as index
                - if loat, then interpretation as %split
            split_before (bool, optional): Splitting data either before (if True) or after (False) 'splitin_at'. Defaults to True.
            plot (bool, optional): Show plot for the training and validation set. Defaults to True.
            plot_size (tuple, optional): Tuple object for plot size (from matplotlib). Defaults to (12, 5).
            legend_loc (str, optional): Set location of the legend in the plot. Defaults to 'upper right'.

        Returns:
            training data, validation data: return training and validation dataset.
        """
        
        # split position: if string, then interpret as Timestamp
        # if int, then interpretation as index
        # if loat, then interpretation as %split

        if isinstance(spliting_at, numbers.Number):
            split_at = spliting_at
        else:
            split_at = pd.Timestamp(spliting_at)
        train, val = data.split_before(split_at)
        if plot:
            plt.figure(101, figsize = plot_size)
            train.plot(label ='training')
            val.plot(label ='validation')
            plt.legend(loc = legend_loc)
        
        return train, val

    def fit_predict(
        self,
        train_data: darts.timeseries.TimeSeries,
        val_data: darts.timeseries.TimeSeries,
        select_model: List[str] = [],
        select_all_models: bool = True,
        seasonality_check: bool = True
    ):
        """Produce accuracy metrics for the selected or all default models.

        Args:
            train_data (darts.timeseries.TimeSeries): Enter darts timeseries data object.
            val_data (darts.timeseries.TimeSeries): Enter darts timeseries data object.
            select_model (List[str]) : Enter list of model names as string. Defaults to [].
            select_all_models (bool) : Select all the available models. Defaults to True.
            seasonality_check (bool) : Call Seasonality_check() method to perform seasonality check. Defaults to True.

        Returns:
            pd.DataFrame: Return accuracy metrics of all selected models.
        """       
        if (not select_model) and (not select_all_models):
            raise ValueError("""
                'select_model' should not be empty list. Select atleast one model or use 'select_all_models' = True !!!
            """)
        
        # elif select_model and select_all_models:
        #   raise ValueError("""
        #     Can not use both 'select_model' and 'select_all_models' = True at the same time. Parameter 'select_all_models' should be set to False !!!
        #   """)
        # else:
        self.train = train_data
        self.val = val_data
        self.seasonality_check(self._data)
        self.selected_models = self.selecting_models(select_model) if select_model else self.selecting_models()
    
        # Evaluate model performance
        def _run_model(
            model_name: str, 
            model
        ):

            pbar.set_description("Processing %s" % model_name)
            t_start =  perf_counter()
            print(f"\n======================={model_name.upper()} - MDOEL SUMMARY=======================")

            print(f"\nModel parameters: {str(model)}")

            # fit the model and compute predictions
            res = model.fit(self.train)
            forecast = model.predict(len(self.val))

            if seasonality_check:
                # for naive forecast, concatenate seasonal fc with drift fc
                if model_name == 'naive drift':
                    if self.is_seasonal:
                        fc_drift = forecast
                        modelS = NaiveSeasonal(K=self.MSEAS)
                        modelS.fit(self.train)
                        fc_seas = modelS.predict(len(self.val))
                        forecast = fc_drift + fc_seas - self.train.last_value()
                res_time = perf_counter() - t_start
            
            print(f"Calculating Error Metrics:..")

            # compute accuracy metrics and processing time
            res_mape = mape(self.val, forecast)
            res_mae = mae(self.val, forecast)
            res_r2 = r2_score(self.val, forecast)
            res_rmse = rmse(self.val, forecast)
            res_rmsle = rmsle(self.val, forecast)
            res_time = perf_counter() - t_start
            res_accuracy = {"MAPE":res_mape, "MAE":res_mae, "R squared":-res_r2, "RMSE":res_rmse, "RMSLE":res_rmsle, "time":res_time}
            results = [forecast, {model_name : res_accuracy}]
            
            print(f"Trial Finished... Total time taken:{res_time} sec")

            return results
    
        pbar = tqdm(self.selected_models.items())
        self._model_predictions = [_run_model(model_name = m_name, model = model) for m_name, model in pbar]
        
        # Prepare Performance Metrics
        res = pd.DataFrame(columns=['MAE', 'MAPE', 'R squared', 'RMSE', 'RMSLE', 'time'])
        for i in range(len(self._model_predictions)):
            res = pd.concat([res, pd.DataFrame(self._model_predictions[i][1]).T])
        pd.set_option("display.precision",3)
        res.style.highlight_min(color="blue", axis=0).highlight_max(color="red", axis=0)
        
        return res
        
    def selecting_models(
        self, 
        model_list: List[str] = None, 
        all: bool = True
    ):
        
        def _get_auto_arima():
            y = np.asarray(self.timeriesdata_to_pdseries(data = self._data))
            # get order of first differencing: the higher of KPSS and ADF test results
            n_kpss = pmd.arima.ndiffs(y, alpha = self.ALPHA, test='kpss', max_d=2)
            n_adf = pmd.arima.ndiffs(y, alpha = self.ALPHA, test='adf', max_d=2)
            n_diff = max(n_adf, n_kpss)

            # get order of seasonal differencing: the higher of OCSB and CH test results
            n_ocsb = pmd.arima.OCSBTest(m=max(4,self.MSEAS)).estimate_seasonal_differencing_term(y)
            n_ch = pmd.arima.CHTest(m=max(4,self.MSEAS)).estimate_seasonal_differencing_term(y)
            ns_diff = max(n_ocsb, n_ch, self.is_seasonal * 1)

            # set up the ARIMA forecaster
            auto_arima_model = AutoARIMA(
                start_p=1, d = n_diff, start_q = 1,
                max_p = 4, max_d = n_diff, max_q = 4,
                start_P = 0, D = ns_diff, start_Q = 0, m = max(4,self.MSEAS), seasonal = self.is_seasonal,
                max_P = 3, max_D = 1, max_Q = 3,
                max_order = 5,                       # p+q+p+Q <= max_order
                stationary = False, 
                information_criterion = "bic", alpha = self.ALPHA, 
                test="kpss", seasonal_test="ocsb",
                stepwise = True, 
                suppress_warnings = True, error_action = "trace", trace = True, with_intercept = "auto")
            return auto_arima_model
        
        def _get_theta():
            # search space for best theta value: check 100 alternatives
            thetas = 2 - np.linspace(-10, 10, 100)

            # initialize search
            best_mape = float('inf')
            best_theta = 0
            # search for best theta among 50 values, as measured by MAPE
            for theta in thetas:
                model = Theta(theta)
                res = model.fit(self.train)
                pred_theta = model.predict(len(self.val))
                res_mape = mape(self.val, pred_theta)

                if res_mape < best_mape:
                    best_mape = res_mape
                    best_theta = theta

            theta_model = Theta(best_theta)   # best theta model among 100
            return theta_model

        _DEFAULT_MODELS = {
            'auto arima' : _get_auto_arima(),
            'exponential smoothing' : ExponentialSmoothing(seasonal_periods= self.MSEAS) if self.is_seasonal else ExponentialSmoothing(),
            'theta' : _get_theta(),
            'naive drift' : NaiveDrift(),
            'prophet (additive seasonality)' : Prophet(seasonality_mode='additive'),
            'prophet (multiplicative seasonality)': Prophet(seasonality_mode='multiplicative')
        }

        if all:
            return _DEFAULT_MODELS
        else:
            filter_model_selection = {k:v for k,v in _DEFAULT_MODELS.items() if k in model_list}
        return filter_model_selection

    def plot_fit_predict(
        self,
    ):
        """Plot the predictions derived by fit_predict().
        """
        # plot the forecasts
        pairs = math.ceil(len(self.selected_models)/2)                    # how many rows of charts
        fig, ax = plt.subplots(pairs, 2, figsize=(20, 5 * pairs))
        ax = ax.ravel()
        
        for i in range(len(self._model_predictions)):
            self.val.plot(label="actual", ax=ax[i])
            self._model_predictions[i][0].plot(label= f"predicted", ax=ax[i], color = "green") 
            mape_model =  list(self._model_predictions[i][1].values())[0]['MAPE']
            time_model =  list(self._model_predictions[i][1].values())[0]['time']
            ax[i].set_title(f"\n\n {list(self._model_predictions[i][1].keys())[0]} : MAPE {mape_model:0.1f}% - time {time_model:0.2f}sec")
        ax[i].set_xlabel("")
        ax[i].legend(loc = 'upper right')
        
    def plot_residual_diagnostics(
        self
    ):
        """Create various visualizations based on models to study residuals.
        """
        # investigate the residuals in the validation dataset
        act = self.val
        df_desc = pd.DataFrame()

        for i in range(len(self._model_predictions)):
            pred = self._model_predictions[i][0]
            resid = pred - act
            df_desc = pd.concat([df_desc, resid.pd_dataframe().describe()], axis=1)
            plot_residuals_analysis(resid);
            plt.title(list(self._model_predictions[i][1].keys())[0])

    def predict(
        self,
        model
    ):
        # TO DO: add prediction functionality
        pass
    
    def save_model(
        self,
        model_list: List[str] = None,
        all: bool = true
    ):
        # TO DO: add save model functionality for more than one model
        pass
    
    @classmethod
    def load_model(
        cls
    ):
        # TO Do: add load model functionality for saved models
        pass
    
    # ====================== TO DO: Additional functionlaities ======================