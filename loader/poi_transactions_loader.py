from pathlib import Path
from matplotlib import pyplot
import numpy as np
import pandas as pd
import json
from tensorflow.keras.models import save_model

class PoiTransactionsLoader:

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

    def df_to_csv(self, output_dir, filename, df):

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir + filename, index_label=False, index=False)

    def save_model_and_weights(self, model, output_dir, n_folds, n_replications):
        output_dir = output_dir + str(n_folds) + "_folds/" + str(n_replications) + "_replications/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        save_model(model, filepath=output_dir)
        #model.save(output_dir+"saved_model")

    def to_json(self, data, dir, filename):

        Path(dir).mkdir(parents=True, exist_ok=True)
        #data = json.dumps(data, ensure_ascii=False)
        with open(dir + filename, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False)

    def save_fig(self, dir, filename, fig):

        Path(dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(dir + filename + ".png",
                    bbox_inches='tight',
                    dpi=400)
