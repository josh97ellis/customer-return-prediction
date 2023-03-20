import pandas as pd

def get_training():
    df = pd.read_csv('data/train.csv').drop(columns='id')
    X = df.drop(columns='return')
    y = df['return']
    return X, y

def return_rate_categories(return_rate):
    if return_rate < 0.25:
        return 'low'
    elif return_rate >= 0.25 and return_rate < 0.50:
        return 'moderately low'
    elif return_rate >= 0.50 and return_rate < 0.75:
        return 'moderately high'
    else:
        return 'high'


def create_customer_return_history() -> None:
    df = pd.read_csv('./data/train.csv')
    
    # Get number of orders and return by customer
    customer_return_history = (
        df
        .groupby('customerID')
        .agg({'return': ['sum', 'count']})
    )
    customer_return_history.columns = ['total_returns', 'total_orders']
    customer_return_history = customer_return_history.reset_index()
    customer_return_history['customerID'] = customer_return_history['customerID'].astype(str)
    
    # Calculate customer return rate
    customer_return_history['return_rate'] = customer_return_history['total_returns'] / customer_return_history['total_orders']
    
    # Get return behavior categories
    customer_return_history['customer_return_behavior'] = customer_return_history['return_rate'].apply(return_rate_categories)
    
    # Write frame to disk
    customer_return_history.to_csv('./data/customer_return_history.csv', index=False)


def create_item_return_history():
    df = pd.read_csv('./data/train.csv')
    item_return_history = df.groupby('itemID').agg({'return': ['sum', 'count']})
    item_return_history.columns = ['total_returns', 'total_orders']
    item_return_history = item_return_history.reset_index()
    item_return_history['itemID'] = item_return_history['itemID'].astype(str)
    item_return_history['return_rate'] = item_return_history['total_returns'] / item_return_history['total_orders']
    item_return_history['item_return_behavior'] = item_return_history['return_rate'].apply(return_rate_categories)
    item_return_history.to_csv('./data/item_return_history.csv', index=False)


def create_manufacturer_return_history():
    df = pd.read_csv('./data/train.csv')
    manufacturer_return_history = df.groupby('manufacturerID').agg({'return': ['sum', 'count']})
    manufacturer_return_history.columns = ['total_returns', 'total_orders']
    manufacturer_return_history = manufacturer_return_history.reset_index()
    manufacturer_return_history['manufacturerID'] = manufacturer_return_history['manufacturerID'].astype(str)
    manufacturer_return_history['return_rate'] = manufacturer_return_history['total_returns'] / manufacturer_return_history['total_orders']
    manufacturer_return_history['manufacturer_return_behavior'] = manufacturer_return_history['return_rate'].apply(return_rate_categories)
    manufacturer_return_history.to_csv('./data/manufacturer_return_history.csv', index=False)


if __name__ == '__main__':
    create_customer_return_history()
    create_item_return_history()
    create_manufacturer_return_history()