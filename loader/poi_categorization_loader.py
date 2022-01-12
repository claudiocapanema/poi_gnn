from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import save_model
import seaborn as sns

class PoiCategorizationLoader:

    def __init__(self):
        pass

    def heatmap(self, dir, df, filename, title, size, annot):

        plt.figure(figsize=size)
        fig = sns.heatmap(df, annot=annot, cmap="YlGnBu").set_title(title).get_figure()

        self.save_fig(dir, filename+".png", fig)

    def save_fig(self, dir, filename, fig):

        Path(dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(dir + filename + ".png",
                    bbox_inches='tight',
                    dpi=400)


    def plot_history_metrics(self, folds_histories, folds_reports, output_dir, show=False):

        n_folds = len(folds_histories)
        n_replications = len(folds_histories[0])
        output_dir = output_dir + str(n_folds) + "_folds/" + str(n_replications) + "_replications/"
        print("pasta: ", output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for i in range(len(folds_histories)):
            fold_histories = folds_histories[i]
            for j in range(len(fold_histories)):
                h = fold_histories[j]
                file_index = "fold_" + str(i) + "_replication_" + str(j)
                plt.figure(figsize=(12, 12))
                plt.plot(h['acc'])
                plt.plot(h['val_acc'])
                plt.title('model acc')
                plt.ylabel('acc')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                if show:
                    plt.show()
                plt.savefig(output_dir + file_index+ "_history_accuracy.png")
                # summarize history for loss
                plt.figure(figsize=(12, 12))
                plt.plot(h['loss'])
                plt.plot(h['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.savefig(output_dir + file_index + "_history_loss.png")
                if show:
                    plt.show()

    def save_report_to_csv(self, output_dir, report, n_folds, n_replications, usuarios):

        precision_dict = {}
        recall_dict = {}
        fscore_dict = {}
        column_size = n_folds*n_replications
        for key in report:
            if key == 'accuracy':
                column = 'accuracy'
                fscore_dict[column] = report[key]
                continue
            elif key == 'recall' or key == 'f1-score' \
                    or key == 'support':
                continue
            if key == 'macro avg' or key == 'weighted avg':
                column = key
                fscore_dict[column] = report[key]['f1-score']
                continue
            fscore_column = key
            fscore_column_data = report[key]['f1-score']
            if len(fscore_column_data) < column_size:
                while len(fscore_column_data) < column_size:
                    fscore_column_data.append(np.nan)
            fscore_dict[fscore_column] = fscore_column_data

            precision_column = key
            precision_column_data = report[key]['precision']
            if len(precision_column_data) < column_size:
                while len(precision_column_data) < column_size:
                    precision_column_data.append(np.nan)
            precision_dict[precision_column] = precision_column_data

            recall_column = key
            recall_column_data = report[key]['recall']
            if len(recall_column_data) < column_size:
                while len(recall_column_data) < column_size:
                    recall_column_data.append(np.nan)
            recall_dict[recall_column] = recall_column_data

            # print("final: ", new_dict)
        precision = pd.DataFrame(precision_dict)
        print("Métricas precision: \n", precision)
        output_dir = output_dir + str(n_folds) + "_folds/" + str(n_replications) + "_replications/"
        print("pasta", output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        precision.to_csv(output_dir + "precision.csv", index_label=False, index=False)

        recall = pd.DataFrame(recall_dict)
        print("Métricas recall: \n", recall)
        recall.to_csv(output_dir + "recall.csv", index_label=False, index=False)

        fscore = pd.DataFrame(fscore_dict)
        print("Métricas fscore: \n", fscore)
        fscore.to_csv(output_dir + "fscore.csv", index_label=False, index=False)

    def save_model_and_weights(self, model, output_dir, n_folds, n_replications):
        output_dir = output_dir + str(n_folds) + "_folds/" + str(n_replications) + "_replications/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        save_model(model, filepath=output_dir)
        #model.save(output_dir+"saved_model")