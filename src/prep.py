from copy import deepcopy
import pandas as pd
import numpy as np


class DataPrep:
    """
    Class to execute the data cleaning and preperations tasks on new data
    """
    def __init__(self):
        pass
    
    def _handle_bad_delivery_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        convert delivery dates of 1990-12-31 as a missing value
        """
        data = deepcopy(df)
        
        # Convert bad dates to NA
        data['deliveryDate'] = (
            np.where(
                data['deliveryDate'] == '1990-12-31',
                np.nan,
                data['deliveryDate']
            )
        )
        
        return data
    
    def _convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Converts objects to Datetime
        '''
        data = deepcopy(df)
        
        # Convert date types to datetime64
        for col in ['orderDate', 'deliveryDate', 'dateOfBirth', 'creationDate']:
            data[col] = pd.to_datetime(data[col], errors='coerce')
        
        return data
    
    def _covert_int_to_string(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Int fields to Str type
        '''
        data = deepcopy(df)
        
        # Cast some int types to string
        for col in ['itemID', 'manufacturerID', 'customerID']:
            data[col] = data[col].astype(str)
        
        return data
    
    def _replace_missing_color(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace missing colors with category = 'No Color'
        """
        data = deepcopy(df)
        data['color'] = data['color'].fillna('No Color')
        return data
    
    def _get_customer_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the customers age at the time of order
        """
        data = deepcopy(df)
        data['customer_age_at_order'] = (
            np.floor(
                (data['orderDate'] - data['dateOfBirth']) / np.timedelta64(1, 'Y')
            )
        )
        return data
    
    def _get_account_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate how long a customer as had an account at the time of order
        """
        data = deepcopy(df)
        data['account_age_months'] = (
            np.floor(
                (data['orderDate'] - data['creationDate']) / np.timedelta64(1, 'M')
            )
        )
        return data
    
    def _get_days_to_deliver(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the number of days it took for the order to be delivered
        """
        data = deepcopy(df)
        data['days_to_deliver'] = (
            np.floor(
                (data['deliveryDate'] - data['orderDate']) / np.timedelta64(1, 'D')
            )
        )
        return data
    
    def _get_order_month(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get the month that the order was placed in
        """
        data = deepcopy(df)
        data['order_month'] = data['orderDate'].dt.month
        return data
    
    def _get_delivered_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        data = deepcopy(df)
        # Determine if the order has been delivered
        data['is_delivered'] = data['deliveryDate'].notnull().astype(int)
        return data
    
    def _remove_lengths_from_pants(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes the length from pant sizes (3432 -> 34)
        """
        data = deepcopy(df)
        
        def trim_last_two(x):
            if len(x) == 4:
                return x[:-2] # slice the string to remove last two characters
            else:
                return x # return the original string if the length is not 4
        
        # apply the function to the column using the apply method and assign it back to the column
        data['size'] = data['size'].apply(trim_last_two)
        data['size'] = data['size'].str.upper().str.replace('+', '', regex=True)
        
        return data

    def _map_size_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts categorical sizes into numerical sizes based on quantile values
        """
        data = deepcopy(df)
        def size_map_category(x):
            x=x.replace("+","").lower()
            if x.isnumeric()==False:  
                if x=='xxxl':
                    x=115
                elif x=='xxl':
                    x=84
                elif x=='xl':
                    x=48
                elif x=='l':
                    x=39
                elif x=='m':
                    x=31
                elif x=='s':
                    x=22
                elif x=='xs':
                    x=10
                else: #for non-reported size we choosing mid value
                    x=38
            return int(x)
        
        data['size'] = data['size'].apply(size_map_category)
        
        return data
    
    def _get_customer_return_rate(self, df: pd.DataFrame):
        data = deepcopy(df)
        
        customer_returns = pd.read_csv('../data/customer_returns.csv', dtype='O')
        customer_returns.rename(
            columns = {
                'total_orders': 'customer_order_count',
                'return_rate': 'customer_return_rate'},
            inplace=True
        )
        
        data = data.merge(
            customer_returns[['customerID', 'customer_return_rate', 'customer_order_count']],
            on='customerID',
            how='left'
        )
        data['customer_order_count'] = pd.to_numeric(data['customer_order_count'], errors='coerce')
        data['customer_return_rate'] = round(pd.to_numeric(data['customer_return_rate'], errors='coerce'), 4)
        return data

    def _get_item_return_rate(self, df: pd.DataFrame):
        data = deepcopy(df)
        
        item_returns = pd.read_csv('../data/item_returns.csv', dtype='O')
        item_returns.rename(
            columns = {
                'total_orders': 'item_order_count',
                'return_rate': 'item_return_rate'},
            inplace=True
        )
        
        data = data.merge(
            item_returns[['itemID', 'item_return_rate']],
            on='itemID',
            how='left'
        )
        
        data['item_return_rate'] = round(pd.to_numeric(data['item_return_rate'], errors='coerce'), 4)
        return data

    def _get_manufacturer_return_rate(self, df: pd.DataFrame):
        data = deepcopy(df)
        
        manufacturer_returns = pd.read_csv('../data/manufacturer_returns.csv', dtype='O')
        manufacturer_returns.rename(
            columns = {
                'total_orders': 'manufacturer_order_count',
                'return_rate': 'manufacturer_return_rate'},
            inplace=True
        )
        
        data = data.merge(
            manufacturer_returns[['manufacturerID', 'manufacturer_return_rate']],
            on='manufacturerID',
            how='left'
        )
        data['manufacturer_return_rate'] = round(pd.to_numeric(data['manufacturer_return_rate'], errors='coerce'), 4)
        return data
    
    
    def run(self, df: pd.DataFrame):
        data_prep = (
            df
            .pipe(self._handle_bad_delivery_dates)
            .pipe(self._convert_dates)
            .pipe(self._covert_int_to_string)
            .pipe(self._replace_missing_color)
            .pipe(self._get_customer_age)
            .pipe(self._get_account_age)
            .pipe(self._get_days_to_deliver)
            .pipe(self._get_order_month)
            .pipe(self._get_delivered_flag)
            .pipe(self._remove_lengths_from_pants)
            .pipe(self._map_size_categories)
            .pipe(self._get_customer_return_rate)
            .pipe(self._get_item_return_rate)
            .pipe(self._get_manufacturer_return_rate)
        )
        
        data_prep = data_prep.drop(
            columns=[
                'orderDate',
                'deliveryDate',
                'creationDate',
                'dateOfBirth',
                'itemID',
                'manufacturerID',
                'customerID'
            ])

        return data_prep