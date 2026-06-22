import pandas as pd

DATA_DIR = "../data/raw/m5-forecasting-accuracy"

def load_sales(stores, category=None, sub_category=None, item_id=None):
    df = pd.read_csv(f"{DATA_DIR}/sales_train_evaluation.csv")
    df = df[df['store_id'].isin(stores)].copy()
    if category:
        df = df[df['cat_id'] == category]
    if sub_category:
        df = df[df['dept_id'] == sub_category]
    if item_id:
        df = df[df['item_id'] == item_id]
    return df

def melt_to_long(sales_wide):
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    value_vars = [c for c in sales_wide.columns if c.startswith('d_')]
    return pd.melt(sales_wide, id_vars=id_vars, value_vars=value_vars,
                   var_name='d', value_name='sales')

def attach_calendar(sales_long, calendar_path):
    calendar = pd.read_csv(calendar_path)
    return pd.merge(sales_long, calendar, on='d', how='left')

def attach_prices(sales_long, prices_path):
    prices = pd.read_csv(prices_path)
    return pd.merge(sales_long, prices,
                    on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

def tidy(sales_long):
    sales_long['date'] = pd.to_datetime(sales_long['date'])
    return sales_long.sort_values(['store_id', 'item_id', 'date']).reset_index(drop=True)

def prepare(stores, category=None):
    """Run the full pipeline end to end and return the tidy long dataframe."""
    sales = load_sales(stores, category)
    long  = melt_to_long(sales)
    long  = attach_calendar(long, f"{DATA_DIR}/calendar.csv")
    long  = attach_prices(long, f"{DATA_DIR}/sell_prices.csv")
    long  = tidy(long)
    return long