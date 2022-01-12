import numpy as np
import pandas as pd

from configuration.poi_categorization_configuration import PoICategorizationConfiguration

def user_category_to_int(user_categories, dataset_name, categories_type):
    category_to_int = PoICategorizationConfiguration().DATASET_CATEGORIES_TO_INT_OSM_CATEGORIES[1][dataset_name][categories_type]

    new_user_categories = []

    for category in user_categories:
        category = str(category)
        converted = category_to_int[category]
        new_user_categories.append(converted)

    return np.array(new_user_categories, dtype=int)

def one_hot_decoding(data):

    new = []
    for e in data:
        new.append(np.argmax(e))

    return new

def one_hot_decoding_predicted(data):

    new = []
    for e in data:
        node_label = []
        for node in e:
            node_label.append(np.argmax(node))
        new.append(node_label)

    new = np.array(new).flatten()
    return new

def top_k_rows(data, k, user_category):

    row_sum = []
    user_unique_categories = {i: 0 for i in pd.Series(user_category).unique().tolist()}
    adjusted_row_sum = []
    for i in range(len(data)):
        row_sum.append([np.sum(data[i]), i, user_category[i]])
        #user_unique_categories[user_category[i]] += 1

    row_sum = sorted(row_sum, reverse=True, key=lambda e:e[0])

    # n_rows_to_remove = len(row_sum) - k
    # count = 0
    # for i in range(len(row_sum) -1, -1, -1):
    #
    #     category = row_sum[i][2]
    #     if user_unique_categories[category] > 1 and count < n_rows_to_remove:
    #         user_unique_categories[category] -= 1
    #         count += 1
    #     else:
    #         adjusted_row_sum.append(row_sum[i])

    #adjusted_row_sum = sorted(adjusted_row_sum, reverse=True, key=lambda e:e[0])


    # if len(row_sum) > k:
    # if row_sum[k][0] < 4:
    #     print("ola")
    #print("Tamanho do row sum: ", len(adjusted_row_sum))
    row_sum = row_sum[:k]
    #print("total: ", adjusted_row_sum)
    row_sum = [e[1] for e in row_sum]

    #print("ids: ", adjusted_row_sum, " tamanho dos dados: ", len(data))

    return np.array(row_sum)

def top_k_rows_category(data, k, user_category):

    row_sum = []
    user_unique_categories = {i: 0 for i in pd.Series(user_category).unique().tolist()}
    adjusted_row_sum = []
    for i in range(len(data)):
        row_sum.append([np.sum(data[i]), i, user_category[i]])
        user_unique_categories[user_category[i]] += 1

    row_sum = sorted(row_sum, reverse=True, key=lambda e:e[0])
    #print("antes: ", row_sum)
    n_rows_to_remove = len(row_sum) - k
    count = 0
    for i in range(len(row_sum) -1, -1, -1):

        category = row_sum[i][2]
        if user_unique_categories[category] > 1 and count < n_rows_to_remove:
            user_unique_categories[category] -= 1
            count += 1
        else:
            adjusted_row_sum.append(row_sum[i])

    #adjusted_row_sum = sorted(adjusted_row_sum, reverse=True, key=lambda e:e[0])


    # if len(row_sum) > k:
    # if row_sum[k][0] < 4:
    #     print("ola")
    #print("Tamanho do row sum: ", len(adjusted_row_sum))
    #row_sum = row_sum[:k]
    #print("total: ", adjusted_row_sum)
    adjusted_row_sum = [e[1] for e in adjusted_row_sum]

    #print("ids: ", adjusted_row_sum, " tamanho dos dados: ", len(data))

    return np.array(adjusted_row_sum)


def filter_data_by_valid_category(user_matrix, user_category, osm_categories):

    idx = []
    print("Tamanho user cate: ",  user_category.shape)
    print("Tamanho user matr: ", user_matrix.shape)
    for i in range(len(user_category)):
        if user_category[i] == "" or user_category[i] == " ":
            continue
        elif user_category[i] not in osm_categories:
            continue
        else:
            idx.append(i)
    idx = np.array(idx)
    if len(idx) == 0:
        return np.array([]), np.array([])
    print(idx)
    user_matrix = user_matrix[idx[:, None], idx]
    user_category = user_category[idx]
    return user_matrix, user_category

def weighted_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask