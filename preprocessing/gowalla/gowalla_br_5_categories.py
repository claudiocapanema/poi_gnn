import pandas as pd



if __name__ == "__main__":

    file = "/media/claudio/Data/backup_win_hd/Downloads/doutorado/gowalla/gowalla_checkins_7_categories_local_datetime_columns_reduced.csv"

    df = pd.read_csv(file)

    df = df.query("country_name == 'Brazil'")

    print(df)

    unique_categories = df['category'].unique().tolist()

    print(unique_categories)

    df.to_csv("/media/claudio/Data/backup_win_hd/Downloads/doutorado/gowalla/gowalla_checkins_5_categories_local_datetime_columns_reduced_br.csv", index=False)