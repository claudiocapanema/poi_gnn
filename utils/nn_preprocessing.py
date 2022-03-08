import numpy as np
import pandas as pd
import networkx as nx

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

import random


def top_k_rows(data, k):

    row_sum = []
    for i in range(len(data)):
        row_sum.append([np.sum(data[i]), i])

    row_sum = sorted(row_sum, reverse=True, key=lambda e:e[0])
    # if len(row_sum) > k:
    # if row_sum[k][0] < 4:
    #     print("ola")
    row_sum = row_sum[:k]

    row_sum = [e[1] for e in row_sum]
    random.seed(1)

    return np.array(row_sum)

def split_graph(data, k, split):

    graph = []

    if data.ndim == 2:
        for i in range(1, split+1):
            matrix = data[k*(i - 1) : k*i, k*(i - 1) : k*i]
            graph.append(matrix)
    else:
        for i in range(1, split+1):
            matrix = data[k*(i - 1) : k*i]
            graph.append(matrix)

    return np.array(graph)

def top_k_rows_category(data, k, user_category):

    row_sum = []
    user_unique_categories = {i: 0 for i in pd.Series(user_category).unique().tolist()}
    categories_weights = {0: 1, 1: 1, 2: 4, 3: 6, 4: 3, 5: 3, 6: 7}
    #categories_weights = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
    adjusted_row_sum = []
    for i in range(len(data)):
        category = user_category[i]
        category_weight = categories_weights[category]
        row_sum.append([np.sum(data[i]), i, category, category_weight])
        user_unique_categories[user_category[i]] += 1

    row_sum = sorted(row_sum, reverse=True, key=lambda e:e[3])
    #print("antes: ", row_sum)
    n_rows_to_remove = len(row_sum) - k
    count = 0

    add = {i: True for i in pd.Series(user_category).unique().tolist()}
    added = []
    while count < k:

        for i in range(len(row_sum)):
            category = row_sum[i][2]

            if add[category] and i not in added and count < k:
                adjusted_row_sum.append(row_sum[i])
                add[category] = False
                added.append(i)
                count += 1

        add = {i: True for i in pd.Series(user_category).unique().tolist()}

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

def top_k_rows_category_user_tracking(data, k, user_category):

    row_sum = []
    user_unique_categories = {i: 0 for i in pd.Series(user_category).unique().tolist()}
    categories_weights = {0: 1, 1: 1, 2: 4, 3: 6, 4: 3, 5: 3, 6: 7, 7:1}
    adjusted_row_sum = []
    for i in range(len(data)):
        category = user_category[i]
        category_weight = categories_weights[category]
        row_sum.append([np.sum(data[i]), i, category, category_weight])
        user_unique_categories[user_category[i]] += 1

    row_sum = sorted(row_sum, reverse=True, key=lambda e:e[3])
    adjusted_row_sum = row_sum[:k]
    adjusted_row_sum = [i[1] for i in adjusted_row_sum]
    #print("antes: ", row_sum)
    # n_rows_to_remove = len(row_sum) - k
    # count = 0
    #
    # add = {i: True for i in pd.Series(user_category).unique().tolist()}
    # added = []
    # while count < k:
    #
    #     for i in range(len(row_sum)):
    #         category = row_sum[i][2]
    #
    #         if add[category] and i not in added and count < k:
    #             adjusted_row_sum.append(row_sum[i])
    #             add[category] = False
    #             added.append(i)
    #             count += 1
    #
    #     add = {i: True for i in pd.Series(user_category).unique().tolist()}
    #
    # #adjusted_row_sum = sorted(adjusted_row_sum, reverse=True, key=lambda e:e[0])
    #
    #
    # # if len(row_sum) > k:
    # # if row_sum[k][0] < 4:
    # #     print("ola")
    # #print("Tamanho do row sum: ", len(adjusted_row_sum))
    # #row_sum = row_sum[:k]
    # #print("total: ", adjusted_row_sum)
    # adjusted_row_sum = [e[1] for e in adjusted_row_sum]

    #print("ids: ", adjusted_row_sum, " tamanho dos dados: ", len(data))

    return np.array(adjusted_row_sum)

def to_networkx(adjacency_matrix):

    new_adjacency_matrix = []
    for i in range(len(adjacency_matrix)):

        for j in range(len(adjacency_matrix)):
            if adjacency_matrix[i][j] != 0:
                new_adjacency_matrix.append((i, j, adjacency_matrix[i][j]))

    g = nx.Graph()
    g.add_weighted_edges_from(new_adjacency_matrix)
    return g

def from_networkx(g):

    new_adjacency_matrix = [[0 for i in range(len(list(g.Nodes)))] for j in range(len(list(g.Nodes)))]

    edges_list = list(g.Edges)

    for i in edges_list:
        from_node = i[0]
        to_node = i[1]
        weight = i[2]
        new_adjacency_matrix[from_node][to_node] = weight

    return new_adjacency_matrix

def top_k_rows_centrality(data, k):

    g = to_networkx(data)
#    centrality = nx.eigenvector_centrality(g, max_iter=10000)
#     centrality = nx.current_flow_betweenness_centrality(g)

    #centrality_list = sorted([(v, float(f"{c:0.2f}")) for v, c in centrality.items()], reverse=True, key=lambda e: e[1])
    nodes_degree = g.degree(list(g.Nodes))
    nodes_degree = sorted(nodes_degree, reverse=True, key= lambda e: e[1])
    idx = [i[0] for i in centrality_list]
    idx = idx[:k]

    return np.array(idx)

def top_k_rows_order(graph, k):

    new_graph = []
    # soma dos pesos de todas as arestas do grafo
    matrix_total = np.array(graph).sum()
    for i in range(len(graph)):

        degree = 0
        row = graph[i]
        # Métrica 1: soma dos pesos das arestas do vértice/PoI "i".
        row_total = sum(row)
        # Métrica 1: ponderar os pesos das arestas do vértice/PoI "i" com base no peso total de todas as arestas do grafo.
        row_total = row_total/matrix_total
        for j in range(len(row)):

            # Métrica 2 : contabilizar o grau do vértice/PoI "i".
            if row[j] != 0:
                degree += 1
        # Métrica 2: grau do vértice/PoI "i" é ponderado com base na quantidade de vértices/PoIs do grafo.
        degree = degree/len(graph)

        # Métrica resultante: aplicar as métricas 1 e 2 na fórmula do f1-score.
        new_graph.append([i, (2*row_total*degree)/(row_total+degree)])

    # Ordenar os PoIs com base na métrica resultante (quanto maior o valor, melhor)
    new_graph = sorted(new_graph, reverse=True, key= lambda e:e[1])
    new_graph = [i[0] for i in new_graph]
    # retorna o grafo com os Top k PoIs.
    new_graph = new_graph[:k]

    return np.array(new_graph)



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