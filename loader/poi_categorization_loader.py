from pathlib import Path
from matplotlib import pyplot
import numpy as np
import pandas as pd
from tensorflow.keras.models import save_model

class PoiCategorizationLoader:

    def __init__(self):
        pass

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
                pyplot.figure(figsize=(12, 12))
                pyplot.plot(h['acc'])
                pyplot.plot(h['val_acc'])
                pyplot.title('model acc')
                pyplot.ylabel('acc')
                pyplot.xlabel('epoch')
                pyplot.legend(['train', 'test'], loc='upper left')
                if show:
                    pyplot.show()
                pyplot.savefig(output_dir + file_index+ "_history_accuracy.png")
                # summarize history for loss
                pyplot.figure(figsize=(12, 12))
                pyplot.plot(h['loss'])
                pyplot.plot(h['val_loss'])
                pyplot.title('model loss')
                pyplot.ylabel('loss')
                pyplot.xlabel('epoch')
                pyplot.legend(['train', 'test'], loc='upper left')
                pyplot.savefig(output_dir + file_index + "_history_loss.png")
                if show:
                    pyplot.show()

    def save_report_to_csv(self, output_dir, report, n_folds, n_replications, usuarios):

        new_dict = {}
        column_size = n_folds*n_replications
        for key in report:
            if key == 'accuracy':
                column = 'accuracy'
                new_dict[column] = report[key]
                continue
            elif key == 'recall' or key == 'f1-score' \
                    or key == 'support':
                continue
            if key == 'macro avg' or key == 'weighted avg':
                column = key.replace(" ", "_") + "_fscore"
                new_dict[column] = report[key]['f1-score']
                continue
            column = key + "_fscore"
            column_data = report[key]['f1-score']
            if len(column_data) < column_size:
                while len(column_data) < column_size:
                    column_data.append(np.nan)
            new_dict[column] = column_data

        #print("final: ", new_dict)
        new_dict['users'] = [str(usuarios)]*column_size
        df = pd.DataFrame(new_dict)
        output_dir = output_dir + str(n_folds) + "_folds/" + str(n_replications) + "_replications/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir + "metrics.csv", index_label=False, index=False)

    def save_model_and_weights(self, model, output_dir, n_folds, n_replications):
        output_dir = output_dir + str(n_folds) + "_folds/" + str(n_replications) + "_replications/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        save_model(model, filepath=output_dir)
        #model.save(output_dir+"saved_model")