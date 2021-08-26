import statistics as st
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, HTML
from extractor.file_extractor import FileExtractor

class PoiCategorizationPerformanceGraphicsLoader:

    def __init__(self):
        self.file_extractor = FileExtractor()

    def _convert_names(self, names):

        convert_dict = {'poi-gnn': 'POI-RGNN', 'arma': 'ARMA'}

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
            model_name_list += [model_name]*len(accuracy)
            accuracy_list += accuracy
            weighted_fscore_list += weighted_fscore

        metrics = pd.DataFrame({'Solution': model_name_list, 'Accuracy': accuracy_list,
                                'Macro f1-score': macro_fscore_list, 'Weighted f1-score': weighted_fscore_list})

        title = ''
        filename = 'barplot_accuracy_ci'
        self.barplot_with_values(metrics, 'Solution', 'Accuracy', base_dir, filename, title)

        #title = 'Macro average fscore'
        filename = 'barplot_macro_avg_fscore_ci'
        self.barplot_with_values(metrics, 'Solution', 'Macro f1-score', base_dir, filename, title)

        #title = 'Weighted average fscore'
        filename = 'barplot_weighted_avg_fscore_ci'
        self.barplot_with_values(metrics, 'Solution', 'Weighted f1-score', base_dir, filename, title)


    def barplot(self, metrics, x_column, y_column, base_dir, file_name, title):
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        plt.figure()
        figure = sns.barplot(x=x_column, y=y_column, data=metrics).set_title(title)
        figure = figure.get_figure()
        figure.savefig(base_dir + file_name + ".png", bbox_inches='tight', dpi=400)

    def barplot_with_values(self, metrics, x_column, y_column, base_dir, file_name, title):
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))
        sns.set(font_scale=1.2, style='whitegrid')
        figure = sns.barplot(x=x_column, y=y_column, data=metrics)

        y_label = "accuracy"
        count = 0
        y_labels = {'macro': [17, 24, 20, 17, 20, 20], 'weighted': [11, 11, 11, 11, 11, 11], 'accuracy': [11, 11, 11, 11, 11, 11]}
        if "macro" in file_name:
            y_label = "macro"
        elif "weighted" in file_name:
            y_label = "weighted"
        for p in figure.patches:
            figure.annotate(format(p.get_height(), '.2f'),
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center',
                             xytext=(0, y_labels[y_label][count]),
                             textcoords='offset points')
            count += 1
        figure = figure.get_figure()
        # plt.legend(bbox_to_anchor=(0.65, 0.74),
        #            borderaxespad=0)
        sorted_values = sorted(metrics[y_column].tolist())
        maximum = sorted_values[-1]
        plt.ylim(0, maximum*1.2)
        # ax.yticks(labels=[df['Precision'].tolist()])
        figure.savefig(base_dir + file_name + ".png", bbox_inches='tight', dpi=400)
        plt.figure()

    def output_dir(self, output_base_dir, dataset_type, category_type, model_name=""):

        return output_base_dir+dataset_type+category_type+model_name

    def export_reports(self, output_dirs, models_names, osm_categories_to_int, base_dir, dataset):

        model_report = {'arma': {}, 'POI-GNN': {}}
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
                precision_means[column] = st.mean(precision[column].tolist())
                recall_means[column] = st.mean(recall[column].tolist())
                fscore_means[column] = st.mean(fscore[column].tolist())

            model_metrics = []

            for column in columns:
                model_metrics.append(precision_means[column])
            for column in columns:
                model_metrics.append(recall_means[column])
            for column in columns:
                model_metrics.append(fscore_means[column])

            models_dict[model_name] = model_metrics


        df = pd.DataFrame(models_dict, index=index).round(2)

        output = base_dir
        Path(output).mkdir(parents=True, exist_ok=True)
        # writer = pd.ExcelWriter(output + 'metrics.xlsx', engine='xlsxwriter')
        #
        # df.to_excel(writer, sheet_name='Sheet1')
        #
        # # Close the Pandas Excel writer and output the Excel file.
        # writer.save()

        max_values = df.idxmax(axis=1)
        max_values = max_values.tolist()
        max_columns = {'arma': [], 'POI-GNN': []}
        for i in range(len(max_values)):
            e = max_values[i]
            max_columns[e].append(i)

        for key in max_columns:
            column_values = df[key].tolist()

            column_list = max_columns[key]
            for j in range(len(column_list)):
                k = column_list[j]
                column_values[k] = "textbf{" + str(column_values[k]) + "}"

            df[key] = np.array(column_values)
        display(HTML(df.to_html()))


        latex = df.to_latex().replace("\}", "}").replace("\{", "{").replace("\\\nRecall", "\\\n\hline\nRecall").replace("\\\nF-score", "\\\n\hline\nF-score")
        pd.DataFrame({'latex': [latex]}).to_csv(output + "latex.txt", header=False, index=False)

        self.plot_general_metrics_with_confidential_interval(model_report, columns, base_dir, dataset)
