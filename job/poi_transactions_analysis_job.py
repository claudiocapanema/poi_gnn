import numpy as np
import json
import pandas as pd
import tensorflow as tf

from domain.user_step_domain import UserStepDomain
from foundation.abs_classes.job import Job
from domain.poi_categorization_domain import PoiCategorizationDomain
from domain.poi_transactions_analysis_domain import PoiTransactionsDomain
from loader.poi_transactions_loader import PoiTransactionsLoader
from extractor.file_extractor import FileExtractor
from foundation.configuration.input import Input
from foundation.util.general_utils import join_df
from configuration.poi_categorization_configuration import PoICategorizationConfiguration
from loader.poi_categorization_loader import PoiCategorizationLoader

class PoiTransactionsAnalysisJob:

    def __init__(self):
        self.user_step_domain = UserStepDomain()
        self.file_extractor = FileExtractor()
        self.poi_categorization_domain = PoiCategorizationDomain(Input.get_instance().inputs['dataset_name'])
        self.poi_transactions_domain = PoiTransactionsDomain()
        self.poi_transactions_loader = PoiTransactionsLoader()

    def start(self):
        base_dir = Input.get_instance().inputs['base_dir']
        number_of_categories = Input.get_instance().inputs['number_of_categories']
        transactions_base_dir = Input.get_instance().inputs['transactions_base_dir']
        transactions_filename = Input.get_instance().inputs['transactions_filename']
        generate_file = Input.get_instance().inputs['generate_file']
        country = Input.get_instance().inputs['country']
        state = Input.get_instance().inputs['state']
        county = Input.get_instance().inputs['county']
        max_interval = Input.get_instance().inputs['max_interval']
        different_venues = Input.get_instance().inputs['different_venues']

        max_interval_dir = ""
        different_venues_dir = ""
        old_8_categories_dir = ""
        if len(max_interval) > 0:
            max_interval_dir = max_interval + "/"
            max_interval = int(max_interval)
        if different_venues == "yes":
            different_venues_dir = "different_venues/"
            different_venues = True

        country_dir = ""
        state_dir = ""
        county_dir = ""
        number_of_categories_dir = number_of_categories + "/"
        transactions_base_dir = transactions_base_dir + number_of_categories_dir
        if country.lower() not in transactions_filename:
            print("Pais diferente")
            raise
        df_transactions = self.poi_transactions_domain.read_file(transactions_filename)
        df_transactions['local_datetime'] = pd.to_datetime(df_transactions['local_datetime'], infer_datetime_format=True)
        print("paises: ", df_transactions['country_code'].unique().tolist())
        # 'userid', 'state', 'county', 'placeid', 'local_datetime', 'latitude',
        #        'longitude', 'category', 'country_code', 'categoryid'

        if len(country) > 0:
            country_dir = country + "/"
            transactions_base_dir = transactions_base_dir + country_dir
        if len(state) > 0:
            print("Estados: ", df_transactions['state'].unique().tolist())
            state_dir = state + "/"
            transactions_base_dir = transactions_base_dir + state_dir
        if len(county) > 0:
            print("county: ", df_transactions['county'].unique().tolist())
            county_dir = county + "/"
            transactions_base_dir = transactions_base_dir + county_dir

        transactions_base_dir = transactions_base_dir + different_venues_dir + max_interval_dir

        if generate_file != "no":
            df_transactions = self.poi_transactions_domain.get_transactions(df_transactions, country, state, county)

            self.poi_transactions_loader.to_json(df_transactions, transactions_base_dir, "transactions.json")
            print("roudou para: ", country, state, county)
            print("configuração: ", max_interval, different_venues, number_of_categories)
        else:
            obs = different_venues_dir + max_interval_dir
            output_dir = "output/transactions/" + number_of_categories_dir + country_dir + state_dir + county_dir + obs
            print("ler", transactions_base_dir)
            df_transactions = self.file_extractor.read_json(transactions_base_dir + "transactions.json")
            self.poi_transactions_domain.statistics_and_plots(output_dir, df_transactions, different_venues, max_interval, country, state, county)
