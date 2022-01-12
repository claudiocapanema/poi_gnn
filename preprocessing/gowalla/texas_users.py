import pandas as pd



if __name__ == "__main__":

    file = "/media/claudio/Data/backup_win_hd/Downloads/doutorado/gowalla/gowalla_checkins_7_categories_local_datetime_columns_reduced_us.csv"

    df = pd.read_csv(file)

    df = df.query("country_name == 'United States' and state_name == 'TEXAS'")

    print(df)

    df.to_csv("/media/claudio/Data/backup_win_hd/Downloads/doutorado/gowalla/gowalla_checkins_7_categories_local_datetime_columns_reduced_us_texas.csv", index=False)