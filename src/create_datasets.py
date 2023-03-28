import pandas as pd


def get_training() -> tuple:
    """
    Read training X and y data
    """
    df = pd.read_csv('../data/train.csv').drop(columns='id')
    X = df.drop(columns='return')
    y = df['return']
    return X, y


def return_rate_categories(return_rate) -> str:
    """
    Conditions for return rates
    """
    if return_rate < 0.25:
        return 'low'
    elif return_rate >= 0.25 and return_rate < 0.50:
        return 'moderately low'
    elif return_rate >= 0.50 and return_rate < 0.75:
        return 'moderately high'
    else:
        return 'high'


def create_customer_return_history() -> None:
    # Read full training data (history of sales)
    df = pd.read_csv('../data/train.csv')
    
    # Get number of orders and return by customer
    df = (
        df
        .groupby('customerID', as_index=False)
        .agg({'return': ['sum', 'count']}))
    
    # Clean up dataframe
    df.columns = ['customerID', 'total_returns', 'total_orders']
    df['customerID'] = df['customerID'].astype(str)
    
    # Calculate customer return rate
    df['return_rate'] = round(df['total_returns'] / df['total_orders'], 4)
    
    # Get return behavior categories
    df['customer_return_category'] = df['return_rate'].apply(return_rate_categories)
    
    # Write frame to disk
    df.to_csv('../data/customer_returns.csv', index=False)


def create_item_return_history():
    # Read full training data (history of sales)
    df = pd.read_csv('../data/train.csv')
    
    # Get number of orders and return by Item
    df = (
        df
        .groupby('itemID', as_index=False)
        .agg({'return': ['sum', 'count']}))
    
    # Clean up dataframe
    df.columns = ['itemID', 'total_returns', 'total_orders']
    df['itemID'] = df['itemID'].astype(str)
    
    # Calculate customer return rate
    df['return_rate'] = round(df['total_returns'] / df['total_orders'], 4)
    
    # Get return behavior categories
    df['item_return_category'] = df['return_rate'].apply(return_rate_categories)
    
    # Write frame to disk
    df.to_csv('../data/item_returns.csv', index=False)


def create_manufacturer_return_history():
    # Read full training data (history of sales)
    df = pd.read_csv('../data/train.csv')
    
    # Get number of orders and return by Item
    df = (
        df
        .groupby('manufacturerID', as_index=False)
        .agg({'return': ['sum', 'count']}))
    
    # Clean up dataframe
    df.columns = ['manufacturerID', 'total_returns', 'total_orders']
    df['manufacturerID'] = df['manufacturerID'].astype(str)
    
    # Calculate customer return rate
    df['return_rate'] = round(df['total_returns'] / df['total_orders'], 4)
    
    # Get return behavior categories
    df['manufacturer_return_category'] = df['return_rate'].apply(return_rate_categories)
    
    # Write frame to disk
    df.to_csv('../data/manufacturer_returns.csv', index=False)
    

def create_customer_abcd_class() -> None:
    df = pd.read_csv('../data/train.csv')
    
    # Group sales by Customer
    customer_abcd = (
        df
        .groupby('customerID', as_index=False)['price'].sum()
        .sort_values(by='price', ascending=False)
        .reset_index(drop=True)
    )
    
    customer_abcd.columns = ['customerID', 'total_sales']
    customer_abcd['total_sales'] = round(customer_abcd['total_sales'], 2)
    
    # Calculate cumulative sales values
    customer_abcd['cumulative_sales'] = round(customer_abcd['total_sales'].cumsum(), 2)
    customer_abcd['cum_perc'] = round(100*customer_abcd['cumulative_sales']/customer_abcd["total_sales"].sum(),2)

    def customer_abcd_mapping(x):
        if x <= 50:
            return 'a'
        elif x > 50 and x <= 75:
            return 'b'
        elif x > 75 and x <= 95:
            return 'c'
        else:
            return 'd'

    # Create customer categories based on sales volumes
    customer_abcd['customer_class'] = customer_abcd['cum_perc'].apply(customer_abcd_mapping)
    customer_abcd.drop(columns=['cumulative_sales', 'cum_perc'], inplace=True)

    customer_abcd.to_csv('../data/customer_abcd.csv', index=False)


if __name__ == '__main__':
    create_customer_return_history()
    create_item_return_history()
    create_manufacturer_return_history()
    create_customer_abcd_class()