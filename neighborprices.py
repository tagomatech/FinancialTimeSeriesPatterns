# neighborprices.py
# tagoma May-23

from blp import blp
import datetime as dt
import numpy as np
import pandas as pd


class NeighborPrices:
    
    def __init__(self, cmdty_name:str, bbg_ticker:str, start_date='2000-01-01') -> pd.DataFrame:
        
        self.cmdty_name = cmdty_name 
        self.bbg_ticker = bbg_ticker
        
        self.bquery = blp.BlpQuery()
        
        self.fields =  ['PX_LAST']
        
        self.start_date = start_date.replace('-', '')
        
        today_dt = dt.datetime.now().date()
        yesterday_dt = today_dt - dt.timedelta(days=1)
        self.end_date = yesterday_dt.strftime('%Y%m%d')
        
        
        #def get_bbg_data(self):
        
        self.bquery.start()
        
        query = blp.create_query(
            request_type='HistoricalDataRequest',
            values={
                'securities' : [self.bbg_ticker],
                'fields' : self.fields,
                'startDate': self.start_date,
                'endDate': self.end_date
            },
        )
        resp = self.bquery.query(query, parse=False, collector=list)

        df = pd.DataFrame.from_records([i['fieldData'] for i in resp[0]['message']['element']['HistoricalDataResponse']['securityData']['fieldData']])
        df = df.assign(Commodity = self.cmdty_name)
        df = df.rename(columns={'date' : 'Date', self.fields[0] : 'Price'})
        df = df[['Commodity', 'Date', 'Price']]
    
        # Compute log returns or simple return in presence of negative prices
        # e.g. WTI futures in April 2020.
        ser = df['Price']
                
        acc = []
        for i in range(1, ser.shape[0]): # Skip the first row
            #print((ser[i-1] > 0) & (ser[i] > 0))
            if ((ser[i-1] > 0) & (ser[i] > 0)):
                #print('higher {}'.format(np.log(ser[i]/ser[i-1] )))
                acc.append(np.log(ser[i]/ser[i-1] ))
            else:
                #print('lower {}'.format(ser[i]/ser[i-1]-1))
                acc.append((ser[i]-ser[i-1])/abs(ser[i-1]))

        ser = pd.Series(data=np.concatenate([[np.nan],  np.array(acc)]), index=ser.index)

        self.neighbor_data = df.assign(Return = ser)


    def get(self):

        return self.neighbor_data

"""
def main():
                                        
    neighbor_data = NeighborPrices('Matif OSR', 'IJ1 Comdty')
    neighbor_data = neighbor_data.get()
    print(neighbor_data)


if __name__ == '__main__':
    main()
"""
