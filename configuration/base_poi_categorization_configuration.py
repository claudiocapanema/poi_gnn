
class BasePoiCategorizationConfiguration:

    def __init__(self):
        self.N_SPLITS = ("n_splits", 5, False, "number of splits (minimum 2) for k fold")

        self.N_REPLICATIONS = (
            "n_replications", 1, False, "number of replications/executions (minimum 1) of training and evaluation process")

        self.DATASET_TYPE = ("dataset_type", {'foursquare': "foursquare/", 'weeplaces': "weeplaces/", 'raw_gps': "raw_gps/",
                                              'global_foursquare': "global_foursquare/", 'dense_foursquare': "dense_foursquare/",
                                              'gowalla': "gowalla/"})

        self.COUNTRY = ("country", {'BR': "BR/", 'US': "US/", 'JP': "JP/", 'Brazil': 'BR/', 'United States': 'US/'})

        self.STATE = ("state", {'sp': "SP/", 'CA': "CA/", 'NY': "NY/", "": "", 'CALIFORNIA': "CA/", "TEXAS": "TX/"})

        self.MAX_TIME_BETWEEN_RECORDS = ("max_time_between_records", {"1": "1_days/", "3": "3_days/", '': ""})

        self.MINIMUM_RECORDS = ("minimum_records", 15)

        self.GRAPH_TYPE = ("graph_type", {'directed':'directed/', 'not_directed': 'not_directed/'})

        self.CATEGORY_TYPE = ("category_type", {'osm': "13_categories/", 'reduced_osm': "9_categories/",
                                                "7_categories": "7_categories/",
                                                "9_categories": "9_categories/",
                                                "8_categories": "8_categories/",
                                                "6_categories": "6_categories/"})

        self.AUGMENTATION_CATEGORIES = ("augmentation_categoires", {0: 6, 6: 3})

        self.DATASET_COLUMNS = ("dataset_columns", {"weeplaces": {"datetime": "local_datetime",
                                              "userid": "userid",
                                              "locationid": "placeid",
                                              "category": "category",
                                              "latitude": "lat",
                                              "longitude": "lon"},
                                "foursquare": {"datetime": "local_datetime",
                                              "userid": "userid",
                                              "locationid": "placeid",
                                              "category": "category",
                                              "latitude": "lat",
                                              "longitude": "lon"},
                                "global_foursquare": {"datetime": "local_datetime",
                                                      "userid": "userid",
                                                      "locationid": "placeid",
                                                      "category": "categoryid",
                                                      "latitude": "latitude",
                                                      "longitude": "longitude"},
                                "gowalla": {"datetime": "local_datetime",
                                                      "userid": "userid",
                                                      "locationid": "placeid",
                                                      "category": "category",
                                                      "latitude": "latitude",
                                                      "longitude": "longitude"},
                                })

        # A categoria Home/hotel foi retirada e englobada por Other.
        # Isso porque o algoritmo já detecta casa.
        # servicos foi removida por não existir categoria equivalente no weeplaces
        self.OSM_7_CATEGORIES_TO_INT = ("osm_categories_to_int",
                                        {'Culture/Tourism': 0,
                                         'Other': 1,
                                         'Gastronomy': 2,
                                         'Transportation': 3,
                                         'Leisure': 4,
                                         'Shopping': 5,
                                         'Educational': 6
                                         })

        self.OSM_9_TO_7_CATEGORIES_TO_INT = ("osm_categories_to_int",
                                        {'Culture/Tourism': 0,
                                         'Home/Hotel': 1,
                                         'Services': 1,
                                         'Gastronomy': 2,
                                         'Transportation': 3,
                                         'Leisure': 4,
                                         'Shopping': 5,
                                         'Educational': 6,
                                         'Other': 1,
                                         'Shop': 5,
                                         'Religious': 1,
                                         'Others': 1,
                                         'Work': 1
                                         })

        self.OSM_9_CATEGORIES_TO_INT = ("osm_categories_to_int",
                                      {'Culture/Tourism': 0,
                                       'Home/Hotel': 1,
                                       'Services': 2,
                                       'Gastronomy': 3,
                                       'Transportation': 4,
                                       'Leisure': 5,
                                       'Shopping': 6,
                                       'Educational': 7,
                                       'Other': 8
                                       }
,
                                        False,
                                        "")

        self.GOWALLA_7_CATEGORIES = ['Shopping', 'Community', 'Food', 'Entertainment', 'Travel', 'Outdoors',
                                     'Nightlife']



        self.FOURSQUARE_CATEGORIES_NAMES_TO_INT_OSM_FIRST_LEVEL_13_CATEGORIES = {'Arts & Crafts Store': 0, 'Home (private)': 1, 'Medical Center': 2, 'Food Truck': 3, 'Food & Drink Shop': 3, 'Coffee Shop': 3, 'Bus Station': 4, 'Bank': 2, 'Gastropub': 5, 'Electronics Store': 6, 'Mobile Phone Shop': 6, 'Café': 3, 'Automotive Shop': 6, 'Restaurant': 3, 'American Restaurant': 3, 'Government Building': 2, 'Airport': 4, 'Office': 2, 'Mexican Restaurant': 3, 'Music Venue': 5, 'Subway': 4, 'Student Center': 2, 'Park': 2, 'Burger Joint': 3, 'Sporting Goods Shop': 6, 'Pizza Place': 3, 'Jewelry Store': 6, 'Sandwich Place': 3, 'Clothing Store': 6, 'Ice Cream Shop': 3, 'Soup Place': 3, 'College Academic Building': 2, 'Department Store': 6, 'Playground': 5, 'Tattoo Parlor': 2, 'Mall': 6, 'University': 7, 'Music Store': 6, 'Salon / Barbershop': 2, 'General College & University': 7, 'Laundry Service': 2, 'Drugstore / Pharmacy': 6, 'Cuban Restaurant': 3, 'Other Nightlife': 5, 'Gym / Fitness Center': 2, 'Italian Restaurant': 3, 'Stadium': 5, 'Church': 8, 'Train Station': 4, 'Tanning Salon': 1, 'Hotel': 2, 'Miscellaneous Shop': 6, 'Bar': 3, 'Spanish Restaurant': 3, 'Asian Restaurant': 3, 'Factory': 9, 'School': 7, 'Burrito Place': 3, 'Fast Food Restaurant': 3, 'Dumpling Restaurant': 3, 'Cupcake Shop': 3, 'Caribbean Restaurant': 3, 'Hardware Store': 6, 'Performing Arts Venue': 0, 'Convenience Store': 6, 'French Restaurant': 3, 'Bookstore': 6, 'Bike Shop': 6, 'Campground': 5, 'Gas Station / Garage': 2, 'Parking': 2, 'Salad Place': 3, 'Art Gallery': 0, 'Video Game Store': 6, 'Toy / Game Store': 6, 'Event Space': 5, 'Vegetarian / Vegan Restaurant': 3, 'Sushi Restaurant': 3, 'Chinese Restaurant': 3, 'Latin American Restaurant': 3, 'Spa / Massage': 5, 'Paper / Office Supplies Store': 6, 'Candy Store': 3, 'Camera Store': 6, 'Breakfast Spot': 3, 'Southern / Soul Food Restaurant': 3, 'Cosmetics Shop': 6, 'Community College': 7, 'Fried Chicken Joint': 3, 'Plaza': 6, 'Dessert Shop': 3, 'Cemetery': 2, 'Museum': 10, 'Bagel Shop': 6, 'Arcade': 5, 'Concert Hall': 5, 'Athletic & Sport': 11, 'Middle Eastern Restaurant': 3, 'Theater': 5, 'Medical School': 7, 'Tea Room': 3, 'Movie Theater': 5, 'Comedy Club': 5, 'Seafood Restaurant': 3, 'Synagogue': 8, 'Donut Shop': 3, 'General Entertainment': 5, 'Pool': 5, 'Japanese Restaurant': 3, 'Arts & Entertainment': 5, 'Pet Store': 6, 'German Restaurant': 3, 'Indian Restaurant': 3, 'Garden': 5, 'Hot Dog Joint': 3, 'Steakhouse': 3, 'Smoke Shop': 5, 'Pool Hall': 5, 'Harbor / Marina': 2, 'Thai Restaurant': 3, 'Bakery': 3, 'Food': 3, 'College Theater': 7, 'Mediterranean Restaurant': 3, 'African Restaurant': 3, 'Outdoors & Recreation': 5, 'Beach': 5, 'Casino': 5, 'Malaysian Restaurant': 3, 'High School': 7, 'Snack Place': 3, 'Taxi': 4, 'College & University': 7, 'Record Shop': 6, 'Temple': 8, 'Historic Site': 10, 'Furniture / Home Store': 6, 'History Museum': 10, 'Bridal Shop': 6, 'Nursery School': 7, 'Antique Shop': 6, 'Taco Place': 3, 'South American Restaurant': 3, 'Law School': 7, 'Thrift / Vintage Store': 6, 'Brazilian Restaurant': 3, 'Winery': 3, 'Greek Restaurant': 3, 'Falafel Restaurant': 3, 'Tapas Restaurant': 3, 'Eastern European Restaurant': 3, 'Korean Restaurant': 3, 'Ski Area': 5, 'Rental Car Location': 4, 'Spiritual Center': 8, 'Science Museum': 10, 'Car Dealership': 6, 'Flea Market': 6, 'Art Museum': 10, 'Gift Shop': 6, 'Portuguese Restaurant': 3, 'Flower Shop': 6, 'Hobby Shop': 6, 'Car Wash': 2, 'Board Shop': 6, 'Cajun / Creole Restaurant': 3, 'Mac & Cheese Joint': 3, 'Shop & Service': 6, 'Vietnamese Restaurant': 3, 'Video Store': 6, 'Travel & Transport': 4, 'Dim Sum Restaurant': 3, 'Racetrack': 5, 'Elementary School': 7, 'Zoo': 5, 'Gaming Cafe': 3, 'Swiss Restaurant': 3, 'Travel Lounge': 4, 'Trade School': 7, 'Australian Restaurant': 3, 'Funeral Home': 2, 'Peruvian Restaurant': 3, 'College Stadium': 7, 'Bike Rental / Bike Share': 4, 'Filipino Restaurant': 3, 'Arepa Restaurant': 3, 'Turkish Restaurant': 3, 'Embassy / Consulate': 2, 'Aquarium': 5, 'Scandinavian Restaurant': 3, 'Middle School': 7, 'Financial or Legal Service': 2, 'Fish & Chips Shop': 6, 'Afghan Restaurant': 3, 'Motorcycle Shop': 6, 'Ethiopian Restaurant': 3, 'Gluten-free Restaurant': 3, 'Argentinian Restaurant': 3, 'Moroccan Restaurant': 3, 'Nightlife Spot': 5, 'Planetarium': 5, 'Storage Facility': 2, 'Molecular Gastronomy Restaurant': 3, 'Internet Cafe': 3, 'Military Base': 12, 'Public Art': 0, 'Market': 6, 'Photography Lab': 2, 'Garden Center': 5, 'Music School': 7, 'Pet Service': 2, 'Rest Area': 12, 'Library': 12, 'Sculpture Garden': 12}

        self.FOURSQUARE_CATEGORIES_NAMES_TO_INT_OSM_FIRST_LEVEL_9_CATEGORIES = {'Arts & Crafts Store': 0, 'Home (private)': 1, 'Medical Center': 2, 'Food Truck': 3, 'Food & Drink Shop': 3, 'Coffee Shop': 3, 'Bus Station': 4, 'Bank': 2, 'Gastropub': 5, 'Electronics Store': 6, 'Mobile Phone Shop': 6, 'Café': 3, 'Automotive Shop': 6, 'Restaurant': 3, 'American Restaurant': 3, 'Government Building': 2, 'Airport': 4, 'Office': 2, 'Mexican Restaurant': 3, 'Music Venue': 5, 'Subway': 4, 'Student Center': 2, 'Park': 2, 'Burger Joint': 3, 'Sporting Goods Shop': 6, 'Pizza Place': 3, 'Jewelry Store': 6, 'Sandwich Place': 3, 'Clothing Store': 6, 'Ice Cream Shop': 3, 'Soup Place': 3, 'College Academic Building': 2, 'Department Store': 6, 'Playground': 5, 'Tattoo Parlor': 2, 'Mall': 6, 'University': 7, 'Music Store': 6, 'Salon / Barbershop': 2, 'General College & University': 2, 'Laundry Service': 2, 'Drugstore / Pharmacy': 6, 'Cuban Restaurant': 3, 'Other Nightlife': 5, 'Gym / Fitness Center': 2, 'Italian Restaurant': 3, 'Stadium': 5, 'Church': 8, 'Train Station': 4, 'Tanning Salon': 1, 'Hotel': 2, 'Miscellaneous Shop': 6, 'Bar': 3, 'Spanish Restaurant': 3, 'Asian Restaurant': 3, 'Factory': 8, 'School': 7, 'Burrito Place': 3, 'Fast Food Restaurant': 3, 'Dumpling Restaurant': 3, 'Cupcake Shop': 3, 'Caribbean Restaurant': 3, 'Hardware Store': 6, 'Performing Arts Venue': 0, 'Convenience Store': 6, 'French Restaurant': 3, 'Bookstore': 6, 'Bike Shop': 6, 'Campground': 5, 'Gas Station / Garage': 2, 'Parking': 2, 'Salad Place': 3, 'Art Gallery': 0, 'Video Game Store': 6, 'Toy / Game Store': 6, 'Event Space': 5, 'Vegetarian / Vegan Restaurant': 3, 'Sushi Restaurant': 3, 'Chinese Restaurant': 3, 'Latin American Restaurant': 3, 'Spa / Massage': 5, 'Paper / Office Supplies Store': 6, 'Candy Store': 3, 'Camera Store': 6, 'Breakfast Spot': 3, 'Southern / Soul Food Restaurant': 3, 'Cosmetics Shop': 6, 'Community College': 7, 'Fried Chicken Joint': 3, 'Plaza': 6, 'Dessert Shop': 3, 'Cemetery': 2, 'Museum': 0, 'Bagel Shop': 6, 'Arcade': 5, 'Concert Hall': 5, 'Athletic & Sport': 5, 'Middle Eastern Restaurant': 3, 'Theater': 5, 'Medical School': 7, 'Tea Room': 3, 'Movie Theater': 5, 'Comedy Club': 5, 'Seafood Restaurant': 3, 'Synagogue': 8, 'Donut Shop': 3, 'General Entertainment': 5, 'Pool': 5, 'Japanese Restaurant': 3, 'Arts & Entertainment': 5, 'Pet Store': 6, 'German Restaurant': 3, 'Indian Restaurant': 3, 'Garden': 5, 'Hot Dog Joint': 3, 'Steakhouse': 3, 'Smoke Shop': 5, 'Pool Hall': 5, 'Harbor / Marina': 2, 'Thai Restaurant': 3, 'Bakery': 3, 'Food': 3, 'College Theater': 7, 'Mediterranean Restaurant': 3, 'African Restaurant': 3, 'Outdoors & Recreation': 5, 'Beach': 5, 'Casino': 5, 'Malaysian Restaurant': 3, 'High School': 7, 'Snack Place': 3, 'Taxi': 4, 'College & University': 7, 'Record Shop': 6, 'Temple': 8, 'Historic Site': 0, 'Furniture / Home Store': 6, 'History Museum': 0, 'Bridal Shop': 6, 'Nursery School': 7, 'Antique Shop': 6, 'Taco Place': 3, 'South American Restaurant': 3, 'Law School': 7, 'Thrift / Vintage Store': 6, 'Brazilian Restaurant': 3, 'Winery': 3, 'Greek Restaurant': 3, 'Falafel Restaurant': 3, 'Tapas Restaurant': 3, 'Eastern European Restaurant': 3, 'Korean Restaurant': 3, 'Ski Area': 5, 'Rental Car Location': 4, 'Spiritual Center': 8, 'Science Museum': 0, 'Car Dealership': 6, 'Flea Market': 6, 'Art Museum': 0, 'Gift Shop': 6, 'Portuguese Restaurant': 3, 'Flower Shop': 6, 'Hobby Shop': 6, 'Car Wash': 2, 'Board Shop': 6, 'Cajun / Creole Restaurant': 3, 'Mac & Cheese Joint': 3, 'Shop & Service': 6, 'Vietnamese Restaurant': 3, 'Video Store': 6, 'Travel & Transport': 4, 'Dim Sum Restaurant': 3, 'Racetrack': 5, 'Elementary School': 7, 'Zoo': 5, 'Gaming Cafe': 3, 'Swiss Restaurant': 3, 'Travel Lounge': 4, 'Trade School': 7, 'Australian Restaurant': 3, 'Funeral Home': 2, 'Peruvian Restaurant': 3, 'College Stadium': 7, 'Bike Rental / Bike Share': 4, 'Filipino Restaurant': 3, 'Arepa Restaurant': 3, 'Turkish Restaurant': 3, 'Embassy / Consulate': 2, 'Aquarium': 5, 'Scandinavian Restaurant': 3, 'Middle School': 7, 'Financial or Legal Service': 2, 'Fish & Chips Shop': 6, 'Afghan Restaurant': 3, 'Motorcycle Shop': 6, 'Ethiopian Restaurant': 3, 'Gluten-free Restaurant': 3, 'Argentinian Restaurant': 3, 'Moroccan Restaurant': 3, 'Nightlife Spot': 5, 'Planetarium': 5, 'Storage Facility': 2, 'Molecular Gastronomy Restaurant': 3, 'Internet Cafe': 3, 'Military Base': 8, 'Public Art': 0, 'Market': 6, 'Photography Lab': 2, 'Garden Center': 5, 'Music School': 7, 'Pet Service': 2, 'Rest Area': 8, 'Library': 8, 'Sculpture Garden': 8}

        self.DATASET_CATEGORIES_TO_INT_OSM_CATEGORIES = ("dataset_categories_to_int_osm_categories",
                                                         {"raw_gps": {"7_categories_osm": self.OSM_9_TO_7_CATEGORIES_TO_INT},
                                                          "foursquare": {"osm": self.FOURSQUARE_CATEGORIES_NAMES_TO_INT_OSM_FIRST_LEVEL_13_CATEGORIES,
                                                                         "reduced_osm": self.FOURSQUARE_CATEGORIES_NAMES_TO_INT_OSM_FIRST_LEVEL_9_CATEGORIES}},
                                                         "")

        self.GLOBAL_FOURSQUARE_8_CATEGORIES = {'Arts & Entertainment': 0,
                                             'Outdoors & Recreation': 1,
                                             'Food': 2,
                                             'Other': 3,
                                             'College & University': 4,
                                             'Shop & Service': 5, 'Travel & Transport': 6, 'Nightlife Spot': 7}

        self.DENSE_FORUSQUARE_9_CATEGORIES = {'Arts and Entertainment': 0,
                                              'Great Outdoors': 1,
                                                'Work': 2,
                                              'Food': 3,
                                              'Travel Spot': 4,
                                              'Home': 5,
                                              'Shop and Service': 6,
                                              'Nighlife Spot': 7,
                                              'College and Education': 8}

        self.GOWALLA_7_CATEGORIES = {'Shopping': 0,
                                     'Community': 1,
                                     'Food': 2,
                                     'Entertainment': 3,
                                     'Travel': 4,
                                     'Outdoors': 5,
                                     'Nightlife': 6}

        self.INT_TO_CATEGORIES = ("int_to_categories", {
            "dense_foursquare": {str(i): list(self.DENSE_FORUSQUARE_9_CATEGORIES.keys())[i] for i in
                                 range(len(list(self.DENSE_FORUSQUARE_9_CATEGORIES.keys())))},
            "gowalla": {str(i): list(self.GOWALLA_7_CATEGORIES.keys())[i] for i in range(len(list(self.GOWALLA_7_CATEGORIES)))}})

        self.CATEGORIES = ("categories", {'weeplaces': {'7_categories_osm': self.OSM_7_CATEGORIES_TO_INT[1]},
                                          'global_foursquare': {'8_categories': self.GLOBAL_FOURSQUARE_8_CATEGORIES},
                                          'dense_foursquare': {'9_categories': self.DENSE_FORUSQUARE_9_CATEGORIES},
                                          'gowalla': {'7_categories': self.GOWALLA_7_CATEGORIES}})

        # MATRIX_MAX_SIZE = ("matrix_max_size", 1166, False, "max size of the adjacency matrix and features matrix")

        self.MAX_SIZE_MATRICES = (
            "max_size_matrices", 10, False, "max size of the adjacency matrices and features (row size) ones")


        self.REPORT_9_INT_CATEGORIES = ("report_9_int_categories",
                                         {'0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '2': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '3': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '4': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '5': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '6': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '7': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '8': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          'accuracy': [],
                                          'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          'weighted avg': {'precision': [], 'recall': [], 'f1-score': [],
                                                           'support': []}},
                                         "report")

        self.REPORT_8_INT_CATEGORIES = ("report_8_int_categories",
                                         {'0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '2': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '3': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '4': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '5': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '6': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '7': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          'accuracy': [],
                                          'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          'weighted avg': {'precision': [], 'recall': [], 'f1-score': [],
                                                           'support': []}},
                                         "report")

        self.REPORT_7_INT_CATEGORIES = ("report_7_int_categories",
                                        {'0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '2': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '3': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '4': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '5': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '6': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         'accuracy': [],
                                         'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         'weighted avg': {'precision': [], 'recall': [], 'f1-score': [],
                                                          'support': []}},
                                        "report")

        self.REPORT_MODEL = ("report_model",
                             {'reduced_osm': self.REPORT_9_INT_CATEGORIES[1],
                              '7_categories_osm': self.REPORT_7_INT_CATEGORIES[1],
                              '7_categories': self.REPORT_7_INT_CATEGORIES[1],
                              '9_categories': self.REPORT_9_INT_CATEGORIES[1],
                              '8_categories': self.REPORT_8_INT_CATEGORIES[1]})

    def output_dir(self, output_base_dir, graph_type, dataset_type, country, category_type, version="", model_name="", state_dir="", max_time_between_records_dir=""):

        return output_base_dir+graph_type+dataset_type+country+state_dir+max_time_between_records_dir+category_type+version+model_name


