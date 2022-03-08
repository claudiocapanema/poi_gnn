import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from configurations import USER_TRACKING_LOCAL_DATETIME_OSM_CATEGORIES, USER_TRACKING_LOCAL_DATETIME_GOWALLA_CATEGORIES_TRANSFER_LEARNING_BR, USER_TRACKING_PREDICTED, USER_TRACKING_HOME_WORK_OTHER_AND_GOWALLA_CATEGORIES

def barplot(df, filename):

    categories_unique = df['poi_resulting'].unique().tolist()
    categories_unique_dict = {i: 0 for i in categories_unique}
    categories = df['poi_resulting'].tolist()
    total = len(df)

    for i in categories:

        categories_unique_dict[i] += 1

    for i in categories_unique_dict:

        categories_unique_dict[i] = categories_unique_dict[i]/total

    df = pd.DataFrame({'Category': list(categories_unique_dict.keys()), 'Percentage': list(categories_unique_dict.values())})

    plt.legend(frameon=False)
    plt.rc('pgf', texsystem='pdflatex')
    plt.figure(dpi=400)
    sns.set(font_scale=1.6, style='whitegrid')
    fig = plt.figure(figsize=(8, 4))
    fig = sns.barplot(x="Percentage", y="Category", data=df, color='cornflowerblue', order=['Home', 'Work', 'Other', 'Food', 'Shopping', 'Community', 'Travel', 'Entertainment', 'Outdoors', 'Nightlife'])
    fig.set_ylabel("")
    fig = fig.get_figure()
    #plt.xticks(rotation=35)
    fig.savefig(filename, bbox_inches='tight')
    fig.savefig(filename.replace("png", "pdf"), bbox_inches='tight')

def count_categories(df):

    uniques = df.unique().tolist()
    uniques_dict = {uniques[i]: 0 for i in range(len(uniques))}

    for i in df:

        uniques_dict[i] += 1

    return pd.DataFrame({'Category': list(uniques_dict.keys()), 'Percentage': list(uniques_dict.values())}).sort_values('Category')


def barplot2(df1, df2, filename):



    sns.set_style()

    #df1['Percentage'] = df1['Percentage'].to_numpy() / total_1
    dataset_version = ['Original'] * len(df1)



    #df2['Percentage'] = df2['Percentage'].to_numpy() / total_2
    dataset_version = dataset_version + ['New'] * len(df2)
    df1 = df1[df1['Category'].isin(
        ['Food', 'Shopping', 'Community', 'Travel', 'Entertainment', 'Outdoors', 'Nightlife'])]
    df2 = df2[df2['Category'].isin(
        ['Food', 'Shopping', 'Community', 'Travel', 'Entertainment', 'Outdoors', 'Nightlife'])]
    categories = df1['Category'].tolist()
    percentage = df1['Percentage'].tolist() + df2['Percentage'].tolist()

    total_1 = df1['Percentage'].to_numpy()

    print("aaaaa")
    print(df2)
    print(df1)
    print("tota")
    print(total_1)
    difference = ((df2['Percentage'].to_numpy() - df1['Percentage'].to_numpy())/total_1)*100

    df = pd.DataFrame({'Category': categories, 'Percentage': difference})

    plt.legend(frameon=False)
    plt.rc('pgf', texsystem='pdflatex')
    plt.figure(dpi=400)
    sns.set(font_scale=1.6, style='whitegrid')
    fig = plt.figure(figsize=(8, 4))
    fig = sns.barplot(x="Percentage", y="Category", data=df, order=['Outdoors','Community','Shopping','Food',   'Travel', 'Entertainment',  'Nightlife'])
    fig.set_ylabel("")
    fig = fig.get_figure()
    #plt.xticks(rotation=35)
    fig.savefig(filename, bbox_inches='tight')
    fig.savefig(filename.replace("png", "svg"), bbox_inches='tight')

    # df = pd.DataFrame({'Category': list(categories_unique_dict_1.keys()) + list(categories_unique_dict_2.keys()), 'Percentage': list(categories_unique_dict_1.values()) + list(categories_unique_dict_2.values()), 'Dataset version':['Original'] * len(list(categories_unique_dict_1.keys())) + ['Enriched'] * len(list(categories_unique_dict_2.keys()))})

    # df = pd.DataFrame(
    #     {'Category': categories, 'Percentage': percentage, 'Dataset version': dataset_version})
    #
    # plt.legend(frameon=False)
    # plt.rc('pgf', texsystem='pdflatex')
    # plt.figure(dpi=400)
    # sns.set(font_scale=1.6, style='whitegrid')
    # fig = plt.figure(figsize=(8, 4))
    # fig = sns.barplot(x="Percentage", y="Category", hue='Dataset version', data=df, order=['Home', 'Work', 'Other', 'Food', 'Shopping', 'Community', 'Travel', 'Entertainment', 'Outdoors', 'Nightlife'])
    # fig.set_ylabel("")
    # fig = fig.get_figure()
    # #plt.xticks(rotation=35)
    # fig.savefig(filename, bbox_inches='tight')
    # fig.savefig(filename.replace("png", "svg"), bbox_inches='tight')

def count(user):

    user = user[['poi_id', 'poi_resulting']].drop_duplicates()
    poi_resulting = user['poi_resulting'].tolist()
    new_poi_resullting = []


    for i in range(len(poi_resulting)):
        poi = poi_resulting[i]
        if poi not in ['Home', 'Work']:
            new_poi_resullting.append(new_poi_resullting)

    return pd.DataFrame({'count': [len(new_poi_resullting)]})


def merge_df(df1, df2):

    df2 = df2[['id', 'poi_id', 'category']].drop_duplicates()

    categories = df1['poi_resulting'].tolist()
    id_list = df1['id'].tolist()
    poi_id_ = df1['poi_id'].tolist()
    new_categories = []

    count = 0
    c_2 = 0

    for i in range(len(df1)):

        user_id = id_list[i]
        poi_id = poi_id_[i]
        category = categories[i]

        if category == 'Other':
            #new_category = df2.query("id == " + str(user_id) + " & poi_id == " + str(poi_id))
            new_category = df2[(df2['id'] == user_id) & (df2['poi_id'] == poi_id)]
            a = df2[(df2['id'] == user_id)]

            # if len(a) > 0:
            #     print("a", a)
            #     print("ids", poi_id)
            if len(new_category) > 0:
                #print("entrou")
                new = new_category['category'].iloc[0]
                if new != 'Other':
                    count += 1
                    new_categories.append(new)
                # if len(new_category['poi_resulting']) > 1:
                #     print("ol")
                #     print(new_category['poi_resulting'])
                #     exit()

            else:

                new_category = df2[df2['poi_id'] == poi_id]['category']

                if len(new_category) > 0:

                    new = new_category.value_counts().idxmax()
                    new_categories.append(new)
                    count += 1

                else:
                    new_categories.append(category)

            c_2 += 1
        else:
            new_categories.append(category)



    df1['poi_resulting'] = np.array(new_categories)
    print("total de others rotulados: ", count/c_2)
    print("Other: ", c_2)

    return df1

if __name__ == "__main__":

    # original data with all categories
    df_1 = pd.read_csv(USER_TRACKING_LOCAL_DATETIME_OSM_CATEGORIES).query("country_name == 'Brazil'")
    df_1 = df_1[df_1['poi_resulting'].isin(['Home', 'Work', 'Other', 'Food', 'Shopping', 'Community', 'Travel', 'Entertainment', 'Outdoors', 'Nightlife'])]
    me = df_1.groupby('id').apply(lambda e: count(e)).query("count > 2").reset_index()
    print(me)
    df_1 = df_1[df_1['id'].isin(me['id'].unique().tolist())]

    print("usuarios bons", len(me))
    print("tamanho antes: ", len(df_1))
    print("usuários antes: ", len(df_1['id'].unique().tolist()))
    print("Pois antes: ", len(df_1[df_1['poi_resulting'].isin(['Other', 'Food', 'Shopping', 'Community', 'Travel', 'Entertainment', 'Outdoors', 'Nightlife'])][['id', 'poi_id']].drop_duplicates()))
    print("df1 value counts: \n", count_categories(df_1[df_1['poi_resulting'].isin(['Other', 'Food', 'Shopping', 'Community', 'Travel', 'Entertainment', 'Outdoors', 'Nightlife'])].drop_duplicates(subset=['id', 'poi_id'])['poi_resulting']))

    df1_value_counts = count_categories(df_1[df_1['poi_resulting'].isin(['Other', 'Food', 'Shopping', 'Community', 'Travel', 'Entertainment', 'Outdoors', 'Nightlife'])].drop_duplicates(subset=['id', 'poi_id'])['poi_resulting'])

    print(len(df_1['id'].unique().tolist()))

    # original data only with the Gowalla categories
    df_2 = pd.read_csv(USER_TRACKING_LOCAL_DATETIME_GOWALLA_CATEGORIES_TRANSFER_LEARNING_BR)

    # predições
    df_3 = pd.read_csv(USER_TRACKING_PREDICTED)

    # print(df_2)
    #
    # print(df_3)

    print("tamanho df3: ", len(df_3))

    print("Value counts predicted: ", df_3.drop_duplicates(subset=['id', 'poi_id'])['category'].value_counts())

    # print("count others 2: ", len(df_2.query("poi_resulting == 'Other'")))



    # df_2 = pd.merge(df_2, df_3, left_on=['id', 'poi_id'], right_on=['id', 'poi_id'])
    # df_2['poi_resulting'] = np.array(df_2['category'].tolist())
    #
    # print("count others 2 novo: ", len(df_2.query("poi_resulting == 'Other'")))

    # print("Merge")
    # print(df_2)
    #
    # print("Merge 3")
    df = merge_df(df_1, df_3)
    print("df value counts: \n", df.drop_duplicates(subset=['id', 'poi_id'])['poi_resulting'].value_counts())
    df_value_counts = count_categories(df[df['poi_resulting'].isin(['Other', 'Food', 'Shopping', 'Community', 'Travel', 'Entertainment', 'Outdoors', 'Nightlife'])].drop_duplicates(subset=['id', 'poi_id'])['poi_resulting'])
    #df_1 = df_1[df_1['id'].isin(df_2['id'].unique().tolist())]
    print("usuários depois: ", len(df_1['id'].unique().tolist()))
    barplot(df_1, "category.png")
    print(df_1['poi_resulting'].unique().tolist())
    barplot(df, "category_new.png")

    print("compara")
    print("antes")
    print(df1_value_counts)
    print("depois")
    print(df_value_counts)

    df.to_csv(USER_TRACKING_HOME_WORK_OTHER_AND_GOWALLA_CATEGORIES, index=False)

    barplot2(df1_value_counts, df_value_counts, "category_comparison.png")