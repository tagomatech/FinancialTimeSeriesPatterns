# neighborchunks.py
# tagoma May-23

from blp import blp
from typing import Union
from sklearn.neighbors import KDTree
import datetime as dt
import numpy as np
import pandas as pd


class NeighborChunks:
    def __init__(self,
                 df:pd.DataFrame,
                 nb_of_chunks:Union[int, None]=None,
                 chunk_length:int=5,
                 nb_neighors:int=5,
                 leaf_size=40,
                 metric='minkowski'):

        self.df = df
        self.ser = df['Return']
        self.ser.index = df['Date']
        
        # Create chunks
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

        #print(ser_in)
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
        
        df_chunks = pd.merge(df_in, df, on='Date', how='inner')
        df_chunks = df_chunks.drop(columns=['Return_x'])
        df_chunks = df_chunks.rename(columns={'Return_y' : 'Return'})
        df_chunks = df_chunks.sort_values(by=['Chunk', 'Feature'])
        
        self.df_chunks = df_chunks.reset_index(drop=True)
        

        # Compute neighbor
        X = df_chunks[df_chunks['Chunk'] < nb_of_chunks].pivot(columns=['Feature'], index='Chunk',  values='Return').values
        y = df_chunks[df_chunks['Chunk'] == nb_of_chunks].pivot(columns=['Feature'], index='Chunk',  values='Return').values

        kdt = KDTree(X, leaf_size=leaf_size, metric=metric)

        def get_indices(idx, chunk_length):
            idx_min = chunk_length*idx#+chunk_length-1
            return (idx_min, idx_min+chunk_length-1)

        dist, idx  = kdt.query(y, k=nb_neighors, return_distance=True)

        idx = [y for x in idx for y in x]
        #idx = [y for x in idx for y in x]
        print(idx)
        if len(idx) == 1:
            idx = idx[0]
            indices = indices(idx, chunk_length)
        else:
            indices = []
            for i in idx:
                indices.append(get_indices(i, chunk_length))

        df_chunks = df_chunks.assign(Neighbor = np.nan)

        n = 1
        for i in indices:
            df_chunks.loc[(df_chunks.index >= i[0]) & (df_chunks.index <= i[1]), 'Neighbor'] = str(n)
            n += 1

        df_chunks.loc[df_chunks['Neighbor'].isna(), 'Neighbor'] = 'target'

        self.df_chunks = df_chunks#print(indices(idx, chunk_length))

    
    def get(self):
        return self.df_chunks  
