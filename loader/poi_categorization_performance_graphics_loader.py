import statistics as st
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, HTML
from extractor.file_extractor import FileExtractor
from foundation.util.statistics_utils import t_distribution_test

class PoiCategorizationPerformanceGraphicsLoader:

    def __init__(self):
        self.file_extractor = FileExtractor()

    def _convert_names(self, names):

        convert_dict = {'poi-gnn': 'POI-RGNN', 'arma': 'ARMA', 'hmrm': 'HMRM'}

        for i in range(len(names)):

            names[i] = convert_dict[names[i]]

        return names

    def plot_metrics(self, metrics, osm_categories_to_int, base_dir, folds_replications_filename):

        title = 'Accuracy'
        filename = folds_replications_filename + '_barplot_accuracy'
        self.barplot_with_values(metrics, 'Method', 'accuracy', base_dir, filename, title)

        title = 'Macro average fscore'
        filename = folds_replications_filename + '_barplot_macro_avg_fscore'
        self.barplot_with_values(metrics, 'Method', 'macro_avg_fscore', base_dir, filename, title)

        title = 'Weighted average fscore'
        filename = folds_replications_filename + '_barplot_weighted_avg_fscore'
        self.barplot_with_values(metrics, 'Method', 'weighted_avg_fscore', base_dir, filename, title)

        columns = list(metrics.columns)

    def plot_general_metrics_with_confidential_interval(self, report, columns, base_dir, dataset):

        sns.set_theme(style='whitegrid')
        macro_fscore_list = []
        model_name_list = []
        accuracy_list = []
        weighted_fscore_list = []
        for model_name in report:

            fscore = report[model_name]['fscore']
            accuracy = fscore['accuracy'].tolist()
            weighted_fscore = fscore['weighted avg'].tolist()
            macro_fscore = fscore['macro avg'].tolist()

            # total_fscore = 0
            # for column in columns:
            #     total_fscore += fscore[column].tolist()
            #
            # macro_fscore = total_fscore/len(columns)
            macro_fscore_list += macro_fscore
            model_name_list += [model_name.upper()]*len(accuracy)
            accuracy_list += accuracy
            weighted_fscore_list += weighted_fscore

        metrics = pd.DataFrame({'Solution': model_name_list, 'Accuracy': accuracy_list,
                                'Macro f1-score': macro_fscore_list, 'Weighted f1-score': weighted_fscore_list})

        sort_order = {'POI-GNN': 1, 'HMRM': 2, 'ARMA': 3}
        metrics['order'] = np.array([sort_order[solution] for solution in metrics['Solution'].tolist()])
        metrics = metrics.sort_values(by='order')
        print("entrou")
        # title = ''
        # filename = 'barplot_accuracy_ci'
        # self.barplot_with_values(metrics, 'Solution', 'Accuracy', base_dir, filename, title)
        #
        # #title = 'Macro average fscore'
        # filename = 'barplot_macro_avg_fscore_ci'
        # self.barplot_with_values(metrics, 'Solution', 'Macro f1-score', base_dir, filename, title)
        #
        # #title = 'Weighted average fscore'
        # filename = 'barplot_weighted_avg_fscore_ci'
        # self.barplot_with_values(metrics, 'Solution', 'Weighted f1-score', base_dir, filename, title)
        #
        # metrics = pd.DataFrame({'Solution': model_name_list, 'Accuracy': accuracy_list,
        #                         'Macro f1-score': macro_fscore_list, 'Weighted f1-score': weighted_fscore_list})

        print(metrics)
        title = ''
        filename = 'barplot_accuracy_ci'
        # mpl.use("pgf")
        # mpl.rcParams.update({
        #     "pgf.texsystem": "pdflatex",
        #     'font.family': 'serif',
        #     'text.usetex': True,
        #     'pgf.rcfonts': False,
        # })
        sns.set(style='whitegrid')

        # fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True, figsize=(14,20), tight_layout=True)
        fig, ax = plt.subplots(ncols=3, nrows=1, sharey=True, sharex=True, figsize=(35, 15), tight_layout=True)

        self.barplot_with_values(metrics, 'Solution', 'Accuracy', base_dir, filename, title, dataset, ax, 2)

        # title = 'Macro average fscore'
        filename = 'barplot_macro_avg_fscore_ci'
        self.barplot_with_values(metrics, 'Solution', 'Macro f1-score', base_dir, filename, title, dataset, ax, 0)

        # title = 'Weighted average fscore'
        filename = 'barplot_weighted_avg_fscore_ci'
        self.barplot_with_values(metrics, 'Solution', 'Weighted f1-score', base_dir, filename, title, dataset, ax, 1)
        # plt.ylim(0, 0.5)
        # plt.tick_params(labelsize=18)
        # plt.grid(True)
        # fig.subplots_adjust(top=0.5, bottom=0, left=0, right=1)

        # fig.tight_layout(pad=1)
        # plt.savefig(dataset + '_metrics_horizontal_latex.pgf')
        print("calculou")
        fig.savefig(base_dir + "_metrics_horizontal.png", bbox_inches='tight', dpi=400)
        fig.savefig(base_dir + "_metrics_horizontal.pdf", bbox_inches='tight', dpi=400)
        fig.savefig(base_dir + "_metrics_horizontal.svg", bbox_inches='tight', dpi=400)
        fig.savefig(base_dir + "_metrics_horizontal.svg", bbox_inches='tight', dpi=400)

        plt.figure()


    def barplot(self, metrics, x_column, y_column, base_dir, file_name, title):
        plt.legend(frameon=False)
        plt.rc('pgf', texsystem='pdflatex')
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        plt.figure()
        figure = sns.barplot(x=x_column, y=y_column, data=metrics).set_title(title)
        plt.figure(dpi=400)
        plt.xticks(rotation=30)
        plt.legend(fontsize=40)
        figure = figure.get_figure()
        figure.savefig(base_dir + file_name + ".png", bbox_inches='tight', dpi=400)
        figure.savefig(base_dir + file_name + ".pdf", bbox_inches='tight', dpi=400)

    def barplot_with_values(self, metrics, x_column, y_column, base_dir, file_name, title, dataset, ax, index):
        plt.legend(frameon=False)
        plt.rc('pgf', texsystem='pdflatex')
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        # plt.figure(figsize=(8, 5))
        # sns.set(font_scale=1.2, style='whitegrid')
        # if y_column == 'Macro f1-score':
        #     order = ['MAP', 'STF-RNN', 'MHSA+PE', 'SERM', 'GARG', 'MFA-RNN']
        # elif y_column == 'Accuracy':
        #     order = ['STF-RNN', 'MAP', 'MHSA+PE', 'SERM', 'GARG', 'MFA-RNN']
        # else:
        #     order = ['MAP', 'STF-RNN', 'MHSA+PE', 'SERM', 'GARG', 'MFA-RNN']
        # order = list(reversed(order))
        # ax[index].set_ylim(0, 0.5)
        size = 35
        sorted_values = sorted(metrics[y_column].tolist())
        maximum = sorted_values[-1]
        if dataset == "users_steps":
            # ax[index].set_ylim(0, maximum * 1.14)
            y_labels = {'macro': [28, 38, 30, 35, 35, 45], 'weighted': [22, 22, 22, 22, 22, 22],
                        'accuracy': [22, 22, 22, 22, 22, 22]}
            # y_labels = {'macro': [22, 22, 22, 22, 22, 22], 'weighted': [22, 22, 22, 22, 22, 22], 'accuracy': [22, 22, 22, 22, 22, 22]}
        else:
            y_labels = {'macro': [30, 30, 30, 30, 30, 30], 'weighted': [30, 30, 30, 30, 30, 30],
                        'accuracy': [30, 30, 30, 30, 30, 30]}
            # ax[index].set_ylim(0, maximum * 1.14)
            if 'weighted' in file_name:
                # ax[index].set_ylim(0, maximum * 1.2)
                pass
        plt.ylim(0, maximum * 1.2)

        # ax[index].set_aspect(5)
        figure = sns.barplot(x=x_column, y=y_column, data=metrics, ax=ax[index])

        figure.set_ylabel(y_column, fontsize=size)
        figure.set_xlabel(x_column, fontsize=size)
        plt.figure(dpi=400)
        plt.xticks(rotation=30)
        plt.legend(fontsize=40)
        # figure0.tick_params(labelsize=10)
        y_label = "accuracy"
        count = 0

        if "macro" in file_name:
            y_label = "macro"
        elif "weighted" in file_name:
            y_label = "weighted"
        for p in figure.patches:
            figure.annotate(format(p.get_height(), '.2f'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            size=size,
                            xytext=(0, y_labels[y_label][count]),
                            textcoords='offset points')
            count += 1
        ax[index].tick_params(axis='x', labelsize=size - 4, rotation=20)
        ax[index].tick_params(axis='y', labelsize=size - 4)



    def output_dir(self, output_base_dir, dataset_type, category_type, model_name=""):

        return output_base_dir+dataset_type+category_type+model_name

    def export_reports(self, output_dirs, models_names, osm_categories_to_int, base_dir, dataset):

        model_report = {'arma': {}, 'POI-GNN': {}, 'hmrm': {}}
        print("saidas: ", models_names)
        for i in range(len(models_names)):
            model_name = models_names[i]
            output = output_dirs[i]
            model_report[model_name]['precision'] = self.file_extractor.read_csv(output + "precision.csv").round(4)
            model_report[model_name]['recall'] = self.file_extractor.read_csv(output + "recall.csv").round(4)
            model_report[model_name]['fscore'] = self.file_extractor.read_csv(output + "fscore.csv").round(4)

        columns = list(osm_categories_to_int.keys())
        index = [np.array(['Precision'] * len(columns) + ['Recall'] * len(columns) + ['Fscore'] * len(columns)), np.array(columns * 3)]
        models_dict = {}
        for model_name in model_report:

            report = model_report[model_name]
            precision = report['precision']
            recall = report['recall']
            fscore = report['fscore']
            precision_means = {}
            recall_means = {}
            fscore_means = {}
            for column in columns:
                precision_means[column] = t_distribution_test(precision[column].tolist())
                recall_means[column] = t_distribution_test(recall[column].tolist())
                fscore_means[column] = t_distribution_test(fscore[column].tolist())

            model_metrics = []

            for column in columns:
                model_metrics.append(precision_means[column])
            for column in columns:
                model_metrics.append(recall_means[column])
            for column in columns:
                model_metrics.append(fscore_means[column])

            models_dict[model_name] = model_metrics


        df = pd.DataFrame(models_dict, index=index).round(4)

        output = base_dir
        print("bbbbbbbbb", base_dir)
        Path(output).mkdir(parents=True, exist_ok=True)
        # writer = pd.ExcelWriter(output + 'metrics.xlsx', engine='xlsxwriter')
        #
        # df.to_excel(writer, sheet_name='Sheet1')
        #
        # # Close the Pandas Excel writer and output the Excel file.
        # writer.save()

        #max_values = df.idxmax(axis=1)
        max_values = self.idmax(df)
        max_columns = {'arma': [], 'POI-GNN': [], 'hmrm': []}
        for max_value in max_values:
            row_index = max_value[0]
            column = max_value[1]
            column_values = df[column].tolist()
            column_values[row_index] = "textbf{" + str(column_values[row_index]) + "}"

            df[column] = np.array(column_values)

        df.columns = ['ARMA', 'POI-GNN', 'HMRM']

        df = df[['POI-GNN', 'HMRM', 'ARMA']]

        display(HTML(df.to_html()))


        latex = df.to_latex().replace("\}", "}").replace("\{", "{").replace("\\\nRecall", "\\\n\hline\nRecall").replace("\\\nF-score", "\\\n\hline\nF-score")
        pd.DataFrame({'latex': [latex]}).to_csv(output + "latex.txt", header=False, index=False)

        self.plot_general_metrics_with_confidential_interval(model_report, columns, base_dir, dataset)

    def idmax(self, df):

        df_indexes = []
        columns = df.columns.tolist()
        print("colunas", columns)
        for i in range(len(df)):

            row = df.iloc[i].tolist()
            indexes = self.select_mean(i, row, columns)
            df_indexes += indexes

        return df_indexes

    def select_mean(self, index, values, columns):

        list_of_means = []
        indexes = []

        for i in range(len(values)):

            value = str(values[i])[:5]
            list_of_means.append(value)

        max_value = max(list_of_means)

        for i in range(len(list_of_means)):

            if list_of_means[i] == max_value:
                indexes.append([index, columns[i]])

        return indexes