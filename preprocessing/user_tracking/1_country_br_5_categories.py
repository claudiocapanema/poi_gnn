import pandas as pd
import numpy as np
from configurations import USER_TRACKING_LOCAL_DATETIME_OSM_CATEGORIES, USER_TRACKING_LOCAL_DATETIME_5_OSM_CATEGORIES_BR

def category_selection(user):

    pass


if __name__ == "__main__":

    df = pd.read_csv(USER_TRACKING_LOCAL_DATETIME_OSM_CATEGORIES)
    print(df)

    df = df.query("poi_resulting in ['Shop', 'Amenity', 'Leisure', 'Tourism', 'Other'] and country_name == 'Brazil'")

    print(df)
    print(df['poi_resulting'].describe())

    unique_categories = df['poi_resulting'].unique().tolist()
    categories_to_int = {unique_categories[i]: i for i in range(len(unique_categories))}
    print("categories to ")
    print(categories_to_int)
    poi_resulting = df['poi_resulting'].tolist()
    poi_resulting_id = np.array([categories_to_int[i] for i in poi_resulting])
    df['poi_resulting_id'] = poi_resulting_id
    print("usuarios únicos: ", len(df['id'].unique().tolist()))

    df.to_csv(USER_TRACKING_LOCAL_DATETIME_5_OSM_CATEGORIES_BR, index=False)
