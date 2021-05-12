'''
timeseries_oversampler.py
================================================================================
The module containing a utility class offering a complete algorithm for 
timeseries oversampling, along with a service function allowing for an immediate 
and simple use of the algorithm.

Copyright 2020 - The IoTwins Project Consortium, Alma Mater Studiorum Università
di Bologna. All rights reserved.
'''
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, math, io, os
from collections import Counter


class ts_oversampler:
    '''
    timeseries_oversampler.ts_oversampler::Class containing the building blocks for 
    the oversampling algorithm, allowing for fine-grained control when needed
    '''
    def generate_new_lengths(self, timeseries, ts_num=1, window_size=6, X=10, fixed_len=True, seed=1):
        ''' 
        timeseries_oversampler.ts_oversampler().generate_new_lengths::Generate the lengths for each of the synthetic timeseries
        
        :param timeseries: list
            timeseries given as a list
        :param ts_num: int
            number of synthetic timeseries lengths to be produced
        :param window_size: int
            network hyperparameters
        :param X: int
            boundary for uniform randomization of the generated lengths
        :param fixed_len: bool
            don't randomize the new lengths if True
        :param seed: int
            random seed
        
        :return: list
            new lengths
        '''
        np.random.seed(seed)
        window_ts_lengths = [len(ts) for ts in timeseries]

        windows = [[] for _ in range(int(max(window_ts_lengths)/window_size) + 1)]
        for ts_len in window_ts_lengths:
            window_pos = int(ts_len/window_size)
            windows[window_pos].append(ts_len)

        # compute the percentage of total timeseries in each window
        tot_ts = len(timeseries)
        prob = [len(window)/tot_ts for window in windows]

        # generate random lengths based on percentages computed above; new_lengths contains pairs in the form: (reference_ts_index_within_window, new_length)
        new_lengths = []
        for rand_window in np.random.choice(list(range(len(prob))), ts_num, p=prob):
            # choose a random reference ts within the chosen time window
            ts_in_window = windows[rand_window]
            reference_ts_pos = np.random.randint(len(ts_in_window))
            new_len = ts_in_window[reference_ts_pos]
            if not fixed_len:
                new_len += np.random.uniform(-X, X)
            # length cannot be lower or higher than this window bounds
            if new_len < rand_window*window_size:
                new_len = rand_window*window_size
            elif new_len >= (rand_window + 1)*window_size:
                new_len = (rand_window + 1)*window_size - 1

            new_lengths.append((reference_ts_pos, int(new_len)))

        return new_lengths

    def random_point_in_d_ball(self, point, radius=-1, seed=1):
        ''' 
        timeseries_oversampler.ts_oversampler().random_point_in_d_ball::Muller algorithm for sampling a random point in a ball of radius d
        
        :param point: list
            point coordinates given as a list
        :param radius: int
            radius in which the point is sampled; if -1, the point will be sampled with a uniform distribution
        :param seed: int
            random seed
        
        :return: list
            sampled point
        '''
        np.random.seed(seed)
        # Muller algorithm
        d = len(point)
        u = np.random.normal(0, 1, d)  # an array of d normally distributed random variables
        norm = np.sum(u**2)**(0.5)
        if radius > -1:
            x = [ax*(point[i]*radius) for i, ax in enumerate(u)]/norm # r*u/norm
        else:
            r = np.random.uniform()**(1.0/d) # radius*np.random.uniform()**(1.0/d)
            x = r * u / norm

        return [x[i]+v for i, v in enumerate(point)]

    def get_centroid(self, points):
        ''' 
        timeseries_oversampler.ts_oversampler().get_centroid::Get the centroid of a list of points
        
        :param points: list
            points for which the centroid has to be found
        
        :return: list
            centroid point
        '''
        centroid = [[] for _ in points[0]]
        l = len(points)

        for point in points:
            for axis, val in enumerate(point):
                centroid[axis].append(val)

        return [sum(values)/l for values in centroid]

    def pad_timeseries(self, timeseries, resulting_length):
        ''' 
        timeseries_oversampler.ts_oversampler().pad_timeseries::Pad the synthetic timeseries to a target length
        
        :param timeseries: list
            timeseries to pad
        :param resulting_length: int
            target length

        :return: list
            padded timeseries
        '''
        if len(timeseries) == resulting_length:
            return timeseries
        iterations = abs(len(timeseries) - resulting_length)
        padded_ts = timeseries
        for i in range(iterations):
            # Replicate first element
            if i % 2 == 0:
                padded_ts = [padded_ts[0]] + padded_ts
            # Replicate last element
            else:
                padded_ts = padded_ts + [padded_ts[-1]]
        return padded_ts
    
    def oversample_timeseries(self, timeseries, window_size=60, ts_num=1, X=8, normal_sd=3.33, sliding_window=3, d=0.01, seed=1):
        ''' 
        timeseries_oversampler.ts_oversampler().oversample_timeseries::Obtain new synthetic timeseries from a list of timeseries
        
        :param timeseries: list
            timeseries to oversample
        :param window_size: int
            window size
        :param ts_num: int
            number of synthetic timeseries to produce
        :param X: int
            boundary for uniform randomization of the generated lengths
        :param normal_sd: int
            standard deviation for normal distribution
        :param sliding_window: int
            sliding window size
        :param d: int
            radius for Muller algorithm
        :param seed: int
            random seed

        :return: list
            synthetic timeseries
        :return: list
            synthetic timeseries lengths
        '''
        np.random.seed(seed)
        new_lengths = sorted(self.generate_new_lengths(timeseries, ts_num=ts_num, window_size=window_size, X=X, seed=seed))
        # sort time series based on timeseries lengths
        timeseries.sort(key=len)

        synthetic_timeseries = []

        for w in range(int(len(timeseries[-1]) / window_size) + 1):
            window_ts = [ts for ts in timeseries if w * window_size <= len(ts) < (w + 1) * window_size] # original timeseries in this window
            window_ts_lengths = [len(ts) for ts in window_ts] # original timeseries lenghts
            window_new_lengths = []
            window_ts_references = []

            for ts_len in new_lengths: # for each new synthetic time series get thos in this window
                if w * window_size <= ts_len[1] < (w + 1) * window_size:
                    window_ts_references.append(ts_len[0])
                    window_new_lengths.append(ts_len[1])

            # skip windows where there are no reference timeseries and any new timeseries to create (second check should be always false if there are no reference ts)
            if len(window_ts) > 0 and len(window_new_lengths) > 0:
                # in the first snapshot for each reference ts get a random neighbour and compute a third point between these two
                first_snapshot_points = [ts[0] for ts in window_ts]

                # ----- COMPUTE NEW TIMESERIES STARTING POINTS -----

                # array with starting points for each new timeseries to create
                starting_points = [None for _ in window_new_lengths]

                # for each reference timeseries assign its first point to its paired synthetic timeseries
                for i, _ in enumerate(window_ts_lengths):
                    for j, pos in enumerate(window_ts_references):
                        if pos == i:
                            starting_points[j] = first_snapshot_points[i]

                # ----- GENERATE POINTS FOR NEW TIMESERIES -----

                # values for new ts based on their reference ts
                generated_points = [[window_ts[window_ts_references[i]][0]] for i, _ in enumerate(window_new_lengths)]
                # value of new ts starting from the chosen starter
                new_ts = [[starting_points[i]] for i, _ in enumerate(window_new_lengths)]

                for snapshot in range(1, len(window_ts[-1])):
                    # all values from all the timeseries which have a value in this position
                    points = [ts[snapshot] for ts in window_ts if len(ts) > snapshot]

                    for ts_pos, ts_length in enumerate(window_new_lengths):
                        if snapshot < ts_length:
                            # reference ts for this new ts
                            reference_ts = window_ts[window_ts_references[ts_pos]]

                            # pick a reference value from reference ts with normal distribution around snapshot (both from past or from future)
                            pos = int(np.random.normal(snapshot, normal_sd, 1)[0])
                            if pos < 0:
                                pos *= -1
                            elif pos >= len(reference_ts):
                                pos -= pos - (len(reference_ts) - 1)
                            reference_ts_value = reference_ts[pos]

                            # sample a point around the randomly chosen one
                            dball_point = self.random_point_in_d_ball(reference_ts_value, d, seed=seed)
                            # add the difference between this new point and the last generated to the actual new ts
                            new_point = [round(new_ts[ts_pos][-1][ax]+(dball_point[ax]-generated_points[ts_pos][-1][ax]), 4) for ax, _ in enumerate(dball_point)]

                            new_ts[ts_pos].append(new_point)
                            generated_points[ts_pos].append(dball_point)

                # ----- MOVING AVERAGE -----
                moving_averages = []
                for j in range(len(new_ts)):
                    moving_averages.append([self.get_centroid(new_ts[j][i-sliding_window:i]) for i in range(sliding_window, len(new_ts[j]))])

                synthetic_timeseries.extend(moving_averages)

        return synthetic_timeseries, new_lengths

    def oversample_and_pad(self, src_data, class_to_oversample, ts_num=1, sl_wnd=3, seed=1):
        ''' 
        timeseries_oversampler.ts_oversampler().oversample_and_pad::Obtain new synthetic timeseries from a list of timeseries
        
        :param src_data: list
            timeseries list to oversample
        :param class_to_oversample: int
            class integer label
        :param ts_num: int
            number of synthetic timeseries to produce
        :param sl_wnd: int
            sliding window size for the oversampling algorithm
        :param seed: int
            random seed

        :return: pandas DataFrame
            synthetic timeseries organized in a table with 'Values' (X) and 'Class' (y) columns
        '''
        synth_ts_list = []
        class_list = []
        
        new_ts, new_len = self.oversample_timeseries(src_data, ts_num=ts_num, sliding_window=sl_wnd, seed=seed)
        
        for i in range(ts_num):   
            padded_ts = self.pad_timeseries(new_ts[i], new_len[i][1])
            synth_ts_list.append(padded_ts)
            class_list.append(class_to_oversample)
            
        data = [synth_ts_list, class_list]
        final_df = pd.DataFrame()
        final_df = final_df.append(data).T
        
        final_df.columns=['Values', 'Class']
        
        return final_df

def gather_and_preprocess_data(_src_df_name, type_labels):
    ''' 
    timeseries_oversampler.gather_and_preprocess_data::Gather, preprocess the
    data and cache it
        
    :param _src_df_name: str
        filename of the csv file containing the raw data

    :return: pandas DataFrame
        preprocessed data in a DataFrame with 'Values' (X) and 'Class' (y)
        columns
    '''

    df = pd.read_csv(_src_df_name)
          
    df['Values'] = df['Values'].apply(lambda x: np.fromstring(x, dtype=float, sep=';'))
    i = 0
    class_i = 0
          
    class_values = pd.DataFrame(data=[df['ID_TimeSeries'], df[type_labels]]).T
    class_values.columns = ['ID_TimeSeries', type_labels]
    data = []
    shapes = []
          
    range_len = len(df['ID_TimeSeries'].unique())
    for k in range(range_len):
        shapes.append(df.iloc[k]['Values'].shape[0])
    
    minimum_shape = min(shapes)


    for _ in range(range_len):
        current_matrix = pd.DataFrame(data=[
            df.iloc[i]['Values'], df.iloc[i+1]['Values'], df.iloc[i+2][
                'Values'], df.iloc[i+3]['Values']]
            ).T.to_numpy()[:minimum_shape,:].tolist()


        data.append([class_i//4, current_matrix, class_values.iloc[class_i][type_labels]])
        i += 4
        class_i += 4


    final_df = pd.DataFrame()
    final_df = final_df.append(data)
    final_df.columns=['ID_TimeSeries', 'Values', 'Class']
    del final_df['ID_TimeSeries']

    return final_df



def augment_ts_dataset(_src_df, _percentage=0, _merge_sets=True, _seed=1):
    ''' 
        timeseries_oversampler.augment_ts_dataset::Obtain a new dataset with synthetic timeseries
        
        :param _src_df: pandas DataFrame
            dataset to augment
        :param _percentage: int
            percentage of augmentation, if set to 0 the service will just balance every class to the one with the highest cardinality
        :param _merge_sets: bool
            if True, the source DataFrame will be merged to the synthetic one
        :param _seed: int
            random seed

        :return: pandas DataFrame
            synthetic timeseries organized in a table with 'Values' (X) and 'Class' (y) columns
        '''
    oversampler = ts_oversampler()
    
    #_src_df = gather_and_preprocess_data(_src_df_name)
    data_list = []

    if _merge_sets:
        data_list.append(_src_df)

    counter = Counter(_src_df['Class'].tolist())
    _percentage += 100
    max_cardinality = counter.most_common()[0][1] * (_percentage/100)   
    num_classes = len(counter)
    
    # Oversample class by class: this allows for keeping a 
    # coherent trend between real and synthetic timeseries wrt each class
    for index in range(num_classes):
        current_class = counter.most_common()[index][0]
        current_aug = round(max_cardinality - counter.most_common()[index][1])
        
        current_class_to_augment_df = _src_df.loc[_src_df['Class'] == int(current_class)]
        
        print("Trying to augment class labeled \"{}\" by {} samples...".format(
            current_class, current_aug))
        current_augmented_class = oversampler.\
            oversample_and_pad(current_class_to_augment_df['Values'].tolist(
                ), current_class, ts_num=current_aug, seed=_seed)
        print("Class labeled \"{}\" augmented!".format(current_class))

        data_list.append(current_augmented_class)

    augmented_df = pd.concat(data_list)
    augmented_df.reset_index()

    # Plot the histogram
    # Before
    fig = plt.figure(figsize=(12,8))
    plt.xlabel('Class', fontsize=20)
    plt.ylabel('Number of Samples', fontsize=20)
    arr = _src_df['Class'].array
    labels, counts = np.unique(arr, return_counts=True)
    plt.bar(labels, counts, align='center', edgecolor="black")
    plt.gca().set_xticks(labels)

    # After
    fig = plt.figure(figsize=(12,8))
    plt.xlabel('Class', fontsize=20)
    plt.ylabel('Number of Samples', fontsize=20)
    arr = augmented_df['Class'].array.astype(int)
    labels, counts = np.unique(arr, return_counts=True)
    plt.bar(labels, counts, align='center', edgecolor="black")
    plt.gca().set_xticks(labels)

    return augmented_df
