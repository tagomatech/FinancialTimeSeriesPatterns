# neighborchunks.py
# tagoma May-23

from blp import blp
from typing import Union
import datetime as dt
import numpy as np
import pandas as pd


class NeighborChunks:
    
    def __init__(self, df:pd.DataFrame, nb_of_chunks:Union[int, None]=None, chunk_length:int=10) -> pd.DataFrame:
   
        self.df = df
        self.ser = df['Return']
        self.ser.index = df['Date']
        ser_in = self.ser.dropna()
        ser_len = ser_in.shape[0]

        if nb_of_chunks == None:
            if chunk_length > ser_len:
                print('Chunks length too large! [return]')
                return
            else:
                ser_in = ser_in[int(ser_len - ((ser_len / chunk_length) // 1 * chunk_length)):]
                nb_of_chunks = int((ser_in.shape[0] / chunk_length) // 1)

        else:
            if nb_of_chunks * chunk_length > ser_len:
                print('Number of chunks * chunks length too large! [return]')
                return
            else:
                ser_in = ser_in[int(ser_len - (chunk_length  * nb_of_chunks)):]

        lst_dat = []
        lst_val = []
        lst_chunk = []
        lst_feat = []

        c = nb_of_chunks
        f = chunk_length
        d = 0

        for i in list(range(ser_in.shape[0], 0, -1)):
            lst_chunk.append(c)
            lst_feat.append(f)
            lst_dat.append(ser_in.index[i - 1 + d])
            lst_val.append(ser_in.values[i - 1 + d])
            f -= 1
            if f == 0:
                c -= 1
                f = chunk_length
                d += chunk_length - 1

        df_in = pd.DataFrame({'Date' : lst_dat,
                              'Return' : lst_val,
                              'Chunk' : lst_chunk,
                              'Feature' : lst_feat,
                             })
        
        df_final = pd.merge(df_in, df, on='Date', how='inner')
        df_final = df_final.drop(columns=['Return_x'])
        df_final = df_final.rename(columns={'Return_y' : 'Return'})
        df_final = df_final.sort_values(by=['Chunk', 'Feature'])
        
        self.df_final = df_final.reset_index(drop=True)

    
    def get(self):
        
        return self.df_final