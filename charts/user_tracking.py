import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

sns.set_style()

def column_as_row(df, metric_name):

    columns = df.columns.tolist()
    metric = []
    category = []
    for column in columns:

        df_column = df[column].tolist()
        size = len(df_column)
        category = category + [column] * size
        metric = metric + df_column

    df = pd.DataFrame({'Category': category, metric_name: metric})

    return df

def bar_plot(df, x, y, filename):
    plt.legend(frameon=False)
    plt.rc('pgf', texsystem='pdflatex')
    figure = sns.barplot(data=df, x=x, y=y)
    plt.figure(dpi=400)
    plt.xticks(rotation=30)
    plt.legend(fontsize=40)
    figure = figure.get_figure()

    figure.savefig(filename, bbox_inches='tight')

    # matplotlib.use("pgf")
    # matplotlib.rcParams.update({
    #     "pgf.texsystem": "pdflatex",
    #     'font.family': 'serif',
    #     'text.usetex': True,
    #     'pgf.rcfonts': False,
    # })
    #
    figure.savefig(filename.replace("png", "svg"), bbox_inches='tight')


if __name__ == "__main__":


    base = "/home/claudio/Documentos/pycharm_projects/poi_gnn/output/poi_categorization_job/base/not_directed/user_tracking/BR/7_categories/5_folds/1_replications/"
    precision = base + "precision.csv"
    recall = base + "recall.csv"
    fscore = base + "fscore.csv"

    df_precision = pd.read_csv(precision)[["Shopping", "Community", "Food", "Entertainment", "Travel", "Outdoors", "Nightlife"]]*100
    df_recall = pd.read_csv(recall)[["Shopping", "Community", "Food", "Entertainment", "Travel", "Outdoors", "Nightlife"]]*100
    df_fscore = pd.read_csv(fscore)[["Shopping", "Community", "Food", "Entertainment", "Travel", "Outdoors", "Nightlife"]]*100

    df_precision = column_as_row(df_precision, 'Precision')
    df_recall = column_as_row(df_recall, 'Recall')
    df_fscore = column_as_row(df_fscore, 'F1-score')

    print(df_precision)

    bar_plot(df_precision, "Category", "Precision", "user_tracking_precision.png")
    bar_plot(df_recall, "Category", "Recall", "user_tracking_recall.png")
    bar_plot(df_fscore, "Category", "F1-score", "user_tracking_fscore.png")
    bar_plot(df_precision, "Category", "Precision", "user_tracking_precision.png")