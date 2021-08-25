from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class PoiCategorizationPerformanceGraphicsLoader:

    def __init__(self):
        pass

    def plot_metrics(self, metrics, osm_categories_to_int, base_dir, folds_replications_filename):

        title = 'Accuracy'
        filename = folds_replications_filename + '_barplot_accuracy'
        self.barplot(metrics, 'Method', 'accuracy', base_dir, filename, title)

        title = 'Macro average fscore'
        filename = folds_replications_filename + '_barplot_macro_avg_fscore'
        self.barplot(metrics, 'Method', 'macro_avg_fscore', base_dir, filename, title)

        title = 'Weighted average fscore'
        filename = folds_replications_filename + '_barplot_weighted_avg_fscore'
        self.barplot(metrics, 'Method', 'weighted_avg_fscore', base_dir, filename, title)

        columns = list(metrics.columns)
        print("antigas: ", columns)
        columns = list(osm_categories_to_int.sub_category()) + [columns[-4], columns[-3], columns[-2], columns[-1]]
        columns = [e.replace("/","_") for e in columns]
        metrics.columns = columns
        metrics.to_csv("metricas_totais.csv", index=False, index_label=False)
        print("novas colunas\n", metrics)
        for i in range(len(list(osm_categories_to_int.sub_category()))):
            title = 'F-score'
            filename = folds_replications_filename + '_barplot_' + columns[i] + "_fscore"
            self.barplot(metrics, 'Method', columns[i], base_dir, filename, title)



    def barplot(self, metrics, x_column, y_column, base_dir, file_name, title):
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        plt.figure()
        figure = sns.barplot(x=x_column, y=y_column, data=metrics).set_title(title)
        figure = figure.get_figure()
        figure.savefig(base_dir + file_name + ".png", bbox_inches='tight', dpi=400)