from copy import deepcopy
import pandas as pd
import numpy as np


class DataPrep:
    def __init__(self):
        pass
    
    def handle_delivery_dates_(self, df: pd.DataFrame):
        """
        convert delivery dates of 1990-12-31 as a missing value
        """
        data = deepcopy(df)
        data['deliveryDate'] = (
            np.where(
                data['deliveryDate'] == '1990-12-31',
                np.nan,
                data['deliveryDate']
            )
        )
        return data
    
    def convert_dates_(self, df: pd.DataFrame):
        '''
        Converts objects to Datetime
        '''
        data = deepcopy(df)
        for col in ['orderDate', 'deliveryDate', 'dateOfBirth', 'creationDate']:
            data[col] = pd.to_datetime(data[col], errors='coerce')
        return data
    
    def covert_int_to_string_(self, df: pd.DataFrame):
        '''
        Int fields to Str type
        '''
        data = deepcopy(df)
        for col in ['itemID', 'manufacturerID', 'customerID']:
            data[col] = data[col].astype(str)
        return data
    
    def replace_missing_color_(self, df: pd.DataFrame):
        """
        Replace missing colors with category = 'No Color'
        """
        data = deepcopy(df)
        data['color'] = data['color'].fillna('No Color')
        return data
    
    def get_customer_age_(self, df: pd.DataFrame):
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
    
    def get_account_age_(self, df: pd.DataFrame):
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
    
    def get_days_to_delivery_(self, df: pd.DataFrame):
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
    
    def get_order_month_(self, df: pd.DataFrame):
        """
        Get the month that the order was placed in
        """
        data = deepcopy(df)
        data['order_month'] = data['orderDate'].dt.month
        return data
    
    def get_delivered_flag_(self, df: pd.DataFrame):
        data = deepcopy(df)
        # Determine if the order has been delivered
        data['is_delivered'] = data['deliveryDate'].notnull().astype(int)
        return data
    
    def remove_lengths_from_pants_(self, df: pd.DataFrame):
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

    def map_size_categories_(self, df: pd.DataFrame):
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
    
    def get_customer_return_behavior_(self, df: pd.DataFrame):
        data = deepcopy(df)
        customer_returns = pd.read_csv('./data/customer_return_history.csv', dtype='O')
        data = data.merge(
            customer_returns[['customerID', 'customer_return_rate', 'customer_order_count']],
            on='customerID',
            how='left'
        )
        data['customer_order_count'] = pd.to_numeric(data['customer_order_count'], errors='coerce')
        data['customer_return_rate'] = round(pd.to_numeric(data['customer_return_rate'], errors='coerce'), 4)
        return data


    def get_item_return_behavior_(self, df: pd.DataFrame):
        data = deepcopy(df)
        customer_returns = pd.read_csv('./data/item_return_history.csv', dtype='O')
        data = data.merge(
            customer_returns[['itemID', 'item_return_rate']],
            on='itemID',
            how='left'
        )
        data['item_return_rate'] = round(pd.to_numeric(data['item_return_rate'], errors='coerce'), 4)
        return data


    def get_manufacturer_return_behavior_(self, df: pd.DataFrame):
        data = deepcopy(df)
        customer_returns = pd.read_csv('./data/manufacturer_return_history.csv', dtype='O')
        data = data.merge(
            customer_returns[['manufacturerID', 'manufacturer_return_rate']],
            on='manufacturerID',
            how='left'
        )
        data['manufacturer_return_rate'] = round(pd.to_numeric(data['manufacturer_return_rate'], errors='coerce'), 4)
        return data
    
    def map_colors_(self, df: pd.DataFrame):
        data = deepcopy(df)
        color_map = pd.read_csv('./data/color_mapping.csv')
        color_map = color_map.set_index('color_key').to_dict()['color_category']
        data = df.replace({"color": color_map})
        return data
    
    def _high_price_mapping(self, df: pd.DataFrame):
        data = deepcopy(df)
        
        data['high_price'] = np.where(data['price'] > 100, 1, 0)
        data.drop(columns='price', inplace=True)
        
        return data
    
    def _map_germany_regions(self, df: pd.DataFrame):
        data = deepcopy(df)
        
        germany_mapping = {
            'North Rhine-Westphalia': 'Central Germany',
            'Lower Saxony': 'Northern Germany',
            'Rhineland-Palatinate': 'Central Germany',
            'Schleswig-Holstein': 'Northern Germany',
            'North Rhine-Westphalia': 'Northern Germany',
            'Hesse': 'Southern Germany',
            'Baden-Wuerttemberg': 'Southern Germany',
            'Bavaria': 'Southern Germany',
            'Berlin': 'Northern Germany',
            'Saxony': 'Central Germany',
            'Brandenburg': 'Southern Germany',
            'Hamburg': 'Northern Germany',
            'Thuringia': 'Central Germany',
            'Mecklenburg-Western Pomerania': 'Northern Germany',
            'Saxony-Anhalt': 'Northern Germany',
            'Bremen': 'Northern Germany',
            'Saarland': 'Central Germany'
        }

        data['region'] = data['state'].replace(germany_mapping)
        data.drop(columns='state', inplace=True)
        
        return data
    
    def run(self, df: pd.DataFrame):
        data_prep = (
            df
            .pipe(self.handle_delivery_dates_)
            .pipe(self.convert_dates_)
            .pipe(self.covert_int_to_string_)
            .pipe(self.replace_missing_color_)
            .pipe(self.get_customer_age_)
            .pipe(self.get_account_age_)
            #.pipe(self.get_days_to_delivery_)
            .pipe(self.get_order_month_)
            .pipe(self.get_delivered_flag_)
            .pipe(self.remove_lengths_from_pants_)
            .pipe(self.map_size_categories_)
            .pipe(self.get_customer_return_behavior_)
            .pipe(self.get_item_return_behavior_)
            .pipe(self.get_manufacturer_return_behavior_)
            #.pipe(self._high_price_mapping)
            #.pipe(self._map_germany_regions)
            #.pipe(self.map_colors_)
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