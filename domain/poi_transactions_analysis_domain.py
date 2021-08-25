import statistics as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ast
from sklearn.model_selection import KFold

import spektral as sk
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import sklearn.metrics as skm
from tensorflow.keras import utils as np_utils

from foundation.util.geospatial_utils import points_distance
from loader.file_loader import FileLoader
from loader.poi_transactions_loader import PoiTransactionsLoader
from extractor.file_extractor import FileExtractor
from model.neural_network.poi_gnn.BR.gnn import GNN
from model.neural_network.poi_gnn.US.gnn import GNNUS
from model.neural_network.poi_gnn.path.gnn import GNNPath
from utils.nn_preprocessing import one_hot_decoding_predicted, top_k_rows


class PoiTransactionsDomain:


    def __init__(self):
        self.file_loader = FileLoader()
        self.file_extractor = FileExtractor()
        self.poi_transations_loader = PoiTransactionsLoader()

    def read_file(self, filename):

        df = self.file_extractor.read_csv(filename)

        return df

    def get_transactions(self, df, country='', state='', county=''):

        if len(country) > 0:
            df = df.query("country_code == '" + country + "'")
        if len(state) > 0:
            df = df.query("state == '" + state + "'")
            print("filt")
            print(df)
        if len(county) > 0:
            if county == "ny":
                df = df.query("county in ['New York', 'Bronx', 'Richmond', 'Queens', 'Kings', 'Manhattan', 'Brooklyn', 'Staten Island']")
            else:
                df = df.query("county == '" + county + "'")

        # 'userid', 'state', 'county', 'placeid', 'local_datetime', 'latitude',
        #        'longitude', 'category', 'country_code', 'categoryid'
        categories = df['category'].unique().tolist()
        #transactions = {categories[i]: {categories[j]: {} for j in range(len(categories))} for i in range(len(categories))}
        transactions_categories = []
        for i in range(len(categories)):

            for j in range(len(categories)):
                transactions_categories.append(categories[i] + "_to_" + categories[j])
        print("antes")
        transactions = {"from_" + categories[i]: {"to_" + categories[j]: [] for j in range(len(categories))} for i in
                        range(len(categories))}
        df.groupby('userid').apply(lambda e: self.user_transcations(e, categories, transactions))
        print("passou")
        self.verify_transactions(transactions)

        return transactions

    def verify_transactions(self, transactions):

        print("verificar")
        from_categories = list(transactions.keys())
        to_categories = list(transactions[from_categories[0]].keys())
        n = 0
        for init in from_categories:

            for end in to_categories:

                if len(transactions[init][end]) == 0:

                    print("errooooooooo")
                    print("sem erros: ", n)
                    exit()
                else:
                    n=+1

        print("Passou verificação")

    def user_transcations(self, user, categories, transactions):

        user = user.sort_values(by='local_datetime')
        week_day_dict = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

        userid = user['userid'].tolist()[0]
        transactions_aggregation_categories = {}
        # for i in range(len(categories)):
        #
        #     for j in range(len(categories)):
        #         transactions_aggregation_categories["from_" + categories[i] + "_to_" + categories[j]] = []

        if len(user['state'].unique().tolist()) > 1:
            #print("Estado diferentes de inicio e fim")
            pass
            #exit()
        if len(user['county'].unique().tolist()) > 1:
            #print("County diferentes de incio e fim")
            pass
            #exit()
        if len(user['country_code'].unique().tolist())>1:
            #print("Paises diferentes de inicio e fim")
            pass
            #exit()
        state = user['state'].to_numpy()
        county = user['county'].to_numpy()
        latitude = user['latitude'].to_numpy()
        longitude = user['longitude'].to_numpy()
        category = user['category'].to_numpy()
        local_datetime = user['local_datetime'].to_numpy()
        week_day = [week_day_dict[e.weekday()] for e in local_datetime]
        hour = [e.hour for e in local_datetime]
        duration_hour = [int((local_datetime[i] - local_datetime[i-1]).total_seconds()/3600) for i in range(1, len(local_datetime))]

        for i in range(1, user.shape[0]-1):
            point_init = [latitude[i-1], longitude[i-1]]
            point_end = [latitude[i], longitude[i]]
            # km
            distance = float(np.round(points_distance(point_init, point_end)/1000, 2))
            transaction = {'userid': userid,
                           'datetime': {'init': str(local_datetime[i-1]), 'end': str(local_datetime[i]),
                                        'init_hour': hour[i-1], 'end_hour': hour[i],
                                        'duration_hour': duration_hour[i], 'week_day_init': week_day[i - 1],
                                        'week_day_end': week_day[i]},
                           'location': {'point_init': point_init, 'point_end': point_end,
                                        'distance_km': distance, 'state_init': state[i-1],
                                        'state_end': state[i],
                                        'county_init': county[i - 1], 'county_end': county[i]}}

            #transactions_aggregation_categories["from_" + category[i-1] + "_to_" + category[i]].append(str(transaction))

            transactions["from_" + category[i - 1]]["to_" + category[i]].append(str(transaction))

        return transactions

    def aggregate_users_transactions(self, df, categories):

        transactions = {"from_" + categories[i]: {"to_" + categories[j]: [] for j in range(len(categories))} for i in
                        range(len(categories))}
        from_categories = df.columns
        for index, row in df.iterrows():

            to_categories = row.index.tolist()

            for i in range(len(from_categories)):

                for j in range(len(to_categories)):

                    transactions[from_categories[i]][to_categories[j]].append(str(row[from_categories[i]].loc[to_categories[j]]))

        return transactions

        #self.category_transactions(df)

    def statistics_and_plots(self, dir, data, different_venues, max_interval, country, state, county):

        from_categories = list(data.keys())
        to_categories = list(data[from_categories[0]].keys())
        if to_categories != list(data[from_categories[1]].keys()):
            print("dif")
            raise

        title = ""
        if len(country) > 0:
            title = title + country + "_"
        if len(state) > 0:
            title = title + state + "_"
        if len(county) > 0:
            title = title + county + "_"
        self.category_transactions(dir, data, different_venues, max_interval, from_categories, to_categories, title)

        # for i in range(len(from_categories)):
        #     from_category = from_categories[i]
        #
        #     for j in range(len(to_categories)):
        #         to_category = to_categories[i]

    def category_transactions(self, dir, data, different_venues, max_interval, from_categories, to_categories, title):

        matrix = []
        print("exemplo")
        print("from_categories")
        print(from_categories)
        print("to_categories")
        print(to_categories)
        transactions = {from_category: {to_category: 0 for to_category in to_categories} for from_category in from_categories}

        # category x category
        transactions_from_list = []
        transactions_to_list = []
        values = []
        # category x category x temporal
        to_hour_dict = {}
        end_week_day_dict = {}
        week_day_init_list = []
        week_day_end_list = []
        distance_list = []
        duration_hour_dict = {}
        distance_km_dict = {}
        duration_hour_list = []
        distance_km_list = []
        total_transactions = 0
        users_dict = {}
        # general plots
        hour_week_day_frequency_dict = {i: 0 for i in range(24)}
        hour_weekend_frequency_dict = {i: 0 for i in range(24)}
        week_day_frequency_dict = {'Monday': 0, 'Tuesday': 0, 'Wednesday': 0, 'Thursday': 0, 'Friday': 0,
                                   'Saturday': 0, 'Sunday': 0}
        category_hour_weekday_frequency_dict = {category: {i: 0 for i in range(24)} for category in to_categories}
        category_hour_weekend_frequency_dict = {category: {i: 0 for i in range(24)} for category in to_categories}
        self.verify_transactions(data)



        for i in range(len(from_categories)):
            from_category = from_categories[i]
            duration_hour_dict[from_category] = {}
            distance_km_dict[from_category] = {}

            for j in range(len(to_categories)):
                to_category = to_categories[j]

                to_hour_dict[from_category+"_"+to_category] = {k: 0 for k in range(24)}
                end_week_day_dict[from_category+"_"+to_category] = {'Monday': 0, 'Tuesday': 0, 'Wednesday': 0,
                                                                    'Thursday': 0, 'Friday': 0, 'Saturday': 0, 'Sunday': 0}
                duration_hour_dict[from_category][to_category] = []
                distance_km_dict[from_category][to_category] = []

                transaction_list = data[from_category][to_category]

                removed_transactions = 0

                # category x category x hour
                for k in range(len(transaction_list)):
                    transaction = ast.literal_eval(transaction_list[k])
                    if different_venues:
                        if float(transaction['location']['distance_km']) == 0:
                            removed_transactions+=1
                            continue
                    if len(str(max_interval)) > 0:
                        if int(transaction['datetime']['duration_hour']) > max_interval:
                            removed_transactions+=1
                            continue
                    #from_hour_list.append(data[from_category][to_category]['datetime']['init_hour'])
                    to_hour_dict[from_category+"_"+to_category][transaction['datetime']['end_hour']]+=1
                    users_dict[transaction['userid']] = 0
                    total_transactions = total_transactions + 1
                    if transaction['datetime']['week_day_end'] not in ['Saturday', 'Sunday']:
                        hour_week_day_frequency_dict[transaction['datetime']['end_hour']]+=1
                        category_hour_weekend_frequency_dict[to_category][transaction['datetime']['end_hour']]+=1
                    else:
                        hour_weekend_frequency_dict[transaction['datetime']['end_hour']]+=1
                        category_hour_weekday_frequency_dict[to_category][transaction['datetime']['end_hour']]+=1
                    week_day_frequency_dict[transaction['datetime']['week_day_end']]+=1
                    duration_hour_dict[from_category][to_category].append(transaction['datetime']['duration_hour'])
                    duration_hour_list.append(transaction['datetime']['duration_hour'])
                    distance_km_list.append(transaction['location']['distance_km'])
                    distance_km_dict[from_category][to_category].append(transaction['location']['distance_km'])
                    end_week_day_dict[from_category+"_"+to_category][transaction['datetime']['week_day_end']]+=1
                    week_day_init_list.append(transaction['datetime']['week_day_init'])
                    week_day_end_list.append(transaction['datetime']['week_day_end'])
                    distance_list.append(transaction['location']['distance_km'])

                # category x category
                transactions_from_list.append(from_category)
                transactions_to_list.append(to_category)
                values.append(len(transaction_list) - removed_transactions)

        df = pd.DataFrame({'from': transactions_from_list, 'to': transactions_to_list, 'total': values})
        df = df.pivot('from', 'to', 'total')

        filename = "categories_heatmap"
        self.generate_plots(dir, df, filename, "categories_" + title)

        ## preprocessing_8_categories
        # hour
        total = []
        from_to_hour = []
        hours = []
        for key in to_hour_dict.keys():

            transaction_hour = to_hour_dict[key]

            for hour in transaction_hour.keys():

                from_to_hour.append(key)
                hours.append(hour)
                total.append(transaction_hour[hour])
        df = pd.DataFrame({'from_to': from_to_hour, 'total': total,
                           'hour': hours})

        df = df.pivot('from_to', 'hour', 'total')
        filename = "categories_hour_heatmap"
        self.generate_plots(dir, df, filename, "categories_hour_" + title, size=(20, 20), annot=False)

        # week day
        total_week_day = []
        from_to_week_day = []
        week_days = []
        for key in end_week_day_dict.keys():

            transaction_week_day = end_week_day_dict[key]

            for hour in transaction_week_day.keys():
                from_to_week_day.append(key)
                week_days.append(hour)
                total_week_day.append(transaction_week_day[hour])

        df = pd.DataFrame({'from_to': from_to_week_day, 'total': total_week_day,
                           'week_day': week_days})

        df = df.pivot('from_to', 'week_day', 'total')[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]

        filename = "categories_week_day_heatmap"
        self.generate_plots(dir, df, filename, "categories_week_day_" + title, size=(20, 20), annot=False)

        self.duration_plots(duration_hour_dict, title, dir)
        self.distance_km_plots(distance_km_dict, title, dir)
        self.week_day_frequency_plot(week_day_frequency_dict, title, dir)
        self.hour_frequency_plot(hour_week_day_frequency_dict, dir, title, "weekday")
        self.hour_frequency_plot(hour_weekend_frequency_dict, dir, title, "weekend")
        self.category_hour_week_day_plot(category_hour_weekday_frequency_dict, dir, title, "weekday")
        self.category_hour_week_day_plot(category_hour_weekend_frequency_dict, dir, title, "weekend")

        df = pd.DataFrame({'total_users': [len(pd.Series(list(users_dict.keys())).unique().tolist())],
                           'total_events': [total_transactions],
                           'average_duration_hour_between_check_ins': [st.mean(duration_hour_list)],
                           'median_duration_hour_between_checkins': [st.median(duration_hour_list)],
                           'average_distance_km_between_checkins': [st.mean(distance_km_list)],
                           'median_distance_km_betwee_checkins': [st.median(distance_km_list)]})

        self.file_loader.save_df_to_csv(df, dir + "merics.csv")


    def generate_plots(self, dir, df, filename, title, size=(10,10), annot=True):


        self.heatmap(dir, df, filename, title, size, annot)

    def duration_plots(self, duration, title, dir):

        average = []
        median = []

        from_categories = list(duration.keys())
        to_categories = list(duration[from_categories[0]].keys())

        from_categories_list = []
        to_categories_list = []

        for from_category in from_categories:

            for to_category in to_categories:
                from_categories_list.append(from_category)
                to_categories_list.append(to_category)
                average.append(round(st.mean(duration[from_category][to_category]), 3))
                median.append(round(st.median(duration[from_category][to_category]), 3))

        df_average = pd.DataFrame({'from': from_categories_list, 'to': to_categories_list, 'average_duration': average})
        df_median = pd.DataFrame({'from': from_categories_list, 'to': to_categories_list, 'median_duration': median})

        df_average = df_average.pivot("from", "to", "average_duration")
        df_median = df_median.pivot("from", "to", "median_duration")

        self.generate_plots(dir, df_average, "heatmap_duration_hours_average" + title, "Duration (hours) average" + title)
        self.generate_plots(dir, df_median, "heatmap_duration_hours_median" + title, "Duration (hours) median" + title)

    def distance_km_plots(self, distance, title, dir):

        from_categories = list(distance.keys())
        to_categories = list(distance[from_categories[0]].keys())

        average = []
        median = []

        from_categories = list(distance.keys())
        to_categories = list(distance[from_categories[0]].keys())

        from_categories_list = []
        to_categories_list = []

        for from_category in from_categories:

            for to_category in to_categories:
                from_categories_list.append(from_category)
                to_categories_list.append(to_category)
                average.append(round(st.mean(distance[from_category][to_category]), 3))
                median.append(round(st.median(distance[from_category][to_category]), 3))

        df_average = pd.DataFrame({'from': from_categories_list, 'to': to_categories_list, 'average_distance_km': average})
        df_median = pd.DataFrame({'from': from_categories_list, 'to': to_categories_list, 'median_distance_km': median})

        df_average = df_average.pivot("from", "to", "average_distance_km")
        df_median = df_median.pivot("from", "to", "median_distance_km")

        self.generate_plots(dir, df_average, "heatmap_distance_km_average" + title, "Average distance (km)" + title)
        self.generate_plots(dir, df_median, "heatmapdistance_km_median" + title, "Median distance (km)" + title)

    def week_day_frequency_plot(self, week_day_frequency_dict, title, dir):

        total = []
        total_frequency = sum(week_day_frequency_dict.values())
        total_frequency = 1
        for day in week_day_frequency_dict:
            total.append(week_day_frequency_dict[day]/total_frequency)
        df = pd.DataFrame({'Day': list(week_day_frequency_dict.keys()), 'Total': total})

        self.barplot(dir, 'Day', 'Total', df, "barplot_week_day_total" + title, "Percentage of events per day" + title)

    def hour_frequency_plot(self, hour_frequency_dict, dir, title, week):

        total = []
        total_frequency = sum(hour_frequency_dict.values())
        total_frequency = 1
        for day in hour_frequency_dict:
            total.append(hour_frequency_dict[day]/total_frequency)
        df = pd.DataFrame({'Hour': list(hour_frequency_dict.keys()), 'Total': total})

        self.barplot(dir, 'Hour', 'Total', df, "barplot_hour_total_" + week + title, "Percentage of events per hour (" + week + ")" + title)

    def category_hour_week_day_plot(self, category_hour_week_day_dict, dir, title, week):

        for to_category in category_hour_week_day_dict:
            total = []
            total_frequency = sum(category_hour_week_day_dict[to_category].values())
            total_frequency = 1
            for day in category_hour_week_day_dict[to_category]:
                total.append(category_hour_week_day_dict[to_category][day] / total_frequency)
            df = pd.DataFrame({'Hour': list(category_hour_week_day_dict[to_category].keys()), 'Total': total})

            self.barplot(dir, 'Hour', 'Total', df,
                         "barplot_category_" + to_category + "_hour_total_" + week + title,
                         "(" + to_category + ") Percentage of events per hour (" + week + ")" + title)

    def heatmap(self, dir, df, filename, title, size, annot):

        plt.figure(figsize=size)
        fig = sns.heatmap(df, annot=annot, cmap="YlGnBu").set_title(title).get_figure()

        self.poi_transations_loader.save_fig(dir, filename, fig)

    def barplot(self, dir, x, y, df, filename, title):

        plt.figure()
        fig = sns.barplot(x=x, y=y, data=df).set_title(title).get_figure()

        self.poi_transations_loader.save_fig(dir, filename, fig)







