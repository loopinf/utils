import re
import warnings
import pandas as pd
import numpy as np
import pandas_gbq
import logging
from exchange_calendars import get_calendar
from types import SimpleNamespace

# import mplfinance as mpf
# import matplotlib.pyplot as plt

# import torch
# import talib
# import torch.nn.functional as F
# from sklearn.preprocessing import MinMaxScaler

class ():
  def __init__(self, date_ref=None, period = 500):
    self.cal_krx = get_calendar('XKRX')
    if date_ref is None:
      date_ref = (pd.Timestamp.today(tz='Asia/Seoul') - cal_krx.day).strftime('%Y-%m-%d')
    # self.dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    self.PROJECT_ID = 'dots-stock'
    self.date_ref = date_ref
    self.period = period
    self.date_map_kernel = {}
    self.date_i_map_kernel = {}
    self.kernel_loaded = {}
    self.path_df_markets = './data/df_markets.parquet'
    self.path_df_price = './data/df_price.parquet'
    self.path_theme = './data/theme.parquet'
    self.path_theme_processed = './data/theme_processed.parquet'
    self.loaded_talib_indica = False

  def get_df_markets(self, date_ref, period = 5):
    # TODO: change to xkrx cal function
    date_ref_ = pd.Timestamp(date_ref).strftime('%Y-%m-%d')
    # date_ref_end = (pd.Timestamp(date_ref) - pd.DateOffset(period, 'D')).strftime('%Y-%m-%d')
    date_ref_end = (pd.Timestamp(date_ref) - self.cal_krx.day * period).strftime('%Y-%m-%d')
    print(f'date_ref_end: {date_ref_end}')
    # Volume is not used in this model
    sql = f'''
      SELECT
        Code, Open, High, Low, Close, ChagesRatio, Dept, Market, date, Name,
        Volume
      FROM
        `dots-stock.red_lion.df_markets_clust_parti`
      WHERE
        date <= "{date_ref_}"
        and date >= "{date_ref_end}"
      ORDER BY
        date
      ''' 
    PROJECT_ID = self.PROJECT_ID
    df = pandas_gbq.read_gbq(sql, project_id=PROJECT_ID, use_bqstorage_api=True)
    df.rename(columns=lambda x: x.lower(), inplace=True) 
    df = df.drop_duplicates(subset=['code','date'], keep='last')
    df.sort_values(['code','date'], inplace=True)
    return df

  def get_price(self, date_ref, period):
    PROJECT_ID = self.PROJECT_ID
    date_ref_ = pd.Timestamp(date_ref).strftime('%Y-%m-%d')
    date_ref_end = (pd.Timestamp(date_ref) - pd.DateOffset(period, 'D')).strftime('%Y-%m-%d')
    table_id = f"{PROJECT_ID}.red_lion.adj_price_{date_ref}"
    logging.debug(f'{date_ref}: {table_id}')
    sql = f'''
      SELECT *
      FROM `{table_id}`
      WHERE
        date <= "{date_ref_}"
        and date >= "{date_ref_end}"
      ORDER BY date
      ''' 
    df = pandas_gbq.read_gbq(sql, project_id=PROJECT_ID, use_bqstorage_api=True)
    df.rename(columns=lambda x: x.lower(), inplace=True) 
    return df

  def load_price(self, date_ref, period):
    self.df_price = self.get_price(date_ref, period)

  def load_markets(self, date_ref=None, period=None):
    if date_ref is None:
      date_ref = self.df_markets.date.max()
    if period is None:
      period = self.period
    self.df_markets = self.get_df_markets(date_ref, period)
    self.name_code = self.df_markets[lambda df: df.date == date_ref][['name', 'code']]
    self.n_c = {name: code for name, code in self.name_code.values}
    self.c_n = {code: name for name, code in self.name_code.values}
    self.nc = SimpleNamespace(**self.n_c) # for dot notation
    self.cn = SimpleNamespace(**self.c_n) # for dot notation

  def get_df_ohocol(self):
    pass

  def get_theme_code(self, date_ref, period):
    PROJECT_ID = self.PROJECT_ID
    date_start = (pd.Timestamp(date_ref) - self.cal_krx.day * period).strftime('%Y%m%d')
    sql = f'''
    SELECT
      *
    FROM
      `dots-stock.stocks_naver.theme_info`
    WHERE
      date_scrape > '{date_start}'
      AND date_scrape <= '{date_ref}'
      '''
    df = pandas_gbq.read_gbq(sql, project_id=PROJECT_ID, use_bqstorage_api=True)
    df.rename(columns=lambda x: x.lower(), inplace=True) 
    df.drop_duplicates(subset=['code','reason_theme','theme_name'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

  def load_df_theme_code(self, date_ref, period=None):
    if period is None:
      period = 10
    self.df_theme_code = self.get_theme_code(date_ref, period=period)

  def _preprocess_theme_name(self, text):
    '''for dataframe'''
    l_remove = ['등']
    text_ = re.sub('\(', '|', text)
    text_ = re.sub('\)', '', text_)
    text_ = text_.strip()
    text_ = re.split('[\|\,]', text_)
    text_ = [word.strip() for word in text_]
    for word in l_remove:
      text_ = [re.sub(word, '', i) for i in text_]
    return text_

  def _get_split_theme(self):
    theme = self.df_theme_code

    theme['theme_name_'] = theme.theme_name.map(self._preprocess_theme_name)
    cols = ['stock_name', 'code', 'theme_name_']
    theme = theme[cols].explode('theme_name_')
    return theme

  def load_df_theme_splited(self):
    if self.df_theme_code is None:
        self.load_df_theme_code(self.df_markets.date.max(), self.period)
    self.df_theme_splited = self._get_split_theme()

  def _save_df_themes_to_local(self, path_theme=None, path_theme_processed=None):
    if path_theme is None:
      path_theme = self.path_theme
    if path_theme_processed is None:
      path_theme_processed = self.path_theme_processed
    self.df_theme_code.to_parquet(path_theme)
    self.df_theme_splited.to_parquet(path_theme_processed)

  def _load_df_themes_from_local(self, path_theme=None, path_theme_processed=None):
    if path_theme is None:
      path_theme = self.path_theme
    if path_theme_processed is None:
      path_theme_processed = self.path_theme_processed
    self.df_theme_code = pd.read_parquet(path_theme)
    self.df_theme_splited = pd.read_parquet(path_theme_processed)
