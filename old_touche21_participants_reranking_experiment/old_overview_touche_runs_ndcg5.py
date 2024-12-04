import pandas as pd
from pathlib import Path
from tabulate import tabulate

#curr_dir = Path(__file__).resolve().parent / "resultsceph" / "best-qrels-all-ndcg5"
curr_dir = Path(__file__).resolve().parent / "resultsceph" / "best-qrels-all-ndcg5-filter"

list_of_dataframes = []


for single_file in curr_dir.iterdir():
    if not single_file.name.startswith('best'):
        continue
    df = pd.read_csv(single_file, index_col="name")
    list_of_dataframes.append(df)

    val = df["nDCG@5 reject"].tolist()
    print(val)


combined_list = []
for x in list_of_dataframes:
    index_names = x.index.tolist()
    group_name = ""
    for name in index_names:
        if not (name.startswith("QSenSim") or name.startswith("QArgSim")):
            group_name = name
    if group_name == "":
        print("No group name")

    x.rename(index={group_name: 'Original'}, inplace=True)

    frame_columns = x.loc[:, ['nDCG@5', 'nDCG@5 reject']]
    frame_columns_transpose = frame_columns.transpose()

    ndcg_frame = frame_columns_transpose.loc[['nDCG@5']]
    ndcg_frame_reject = frame_columns_transpose.loc[['nDCG@5 reject']]

    for col_index, cell_value in ndcg_frame.iloc[0].iteritems():
        reject = ndcg_frame_reject[col_index][0]

        print("Column Index:", col_index)
        print("Cell Value:", cell_value)

        val = ndcg_frame_reject[col_index][0]
        add = ""
        if val != False:
            add = "sig"
        signal = str(cell_value) + add

        ndcg_frame.at['nDCG@5', col_index] = signal
        mewo = 1

    ndcg_frame.rename(index={'nDCG@5': group_name}, inplace=True)


    combined_list.append(ndcg_frame)

combined_df = pd.concat(combined_list, axis=0)

for index, row in combined_df.iterrows():
    best_value = 0
    best_column = ""
    for col in combined_df.columns:
        val = combined_df.at[index,col]
        val_clear = val.replace("sig","")
        val_float = float(val_clear)
        if val_float > best_value:
            best_column = col
            best_value = val_float
    for col in combined_df.columns:
        val = combined_df.at[index, col]

        add = ""
        if val.endswith("sig"):
            add = "$^{\dag}$"

        val_clear = val.replace("sig","")
        val_float = float(val_clear)
        round_cell_value = round(val_float, 3)
        round_cell_value_print = str(round_cell_value)

        if col == best_column:
            round_cell_value_print = r"\underline{" + round_cell_value_print + r"}" + add

        if "Dirichlet" in index:
            round_cell_value_print = (r"\textbf{" + round_cell_value_print + r"}")

        combined_df.at[index, col] = round_cell_value_print



index_names = ["Elrond",
 ("Pippin-Took", "Pippin Took"),
 ("Robin-Hood","Robin Hood"),
 "Asterix",
          ("Dread-Pirate-Roberts","Dread Pirate Roberts",),
 "Skeletor",
          ("Luke-Skywalker","Luke Skywalker",),
 "Shanks",
 "Heimdall",
 "Athos",
          ("Goemon-Ishikawa","Goemon Ishikawa"),
 ("Jean-pierre-polnareff","Jean Piere Polnareff"),
("Baseline-Dirichlet","DirichletLM")
 ]

column_names = [
    ("Original", "O", "Orig."),
    ("QSenSim_max_exact_sbert", "A", r"QS$_{max}^{e}$"),
    ("QSenSim_max_sbert", "B", r"QS$_{max}$"),

    ("QSenSim_mean_exact_sbert", "C", r"QS$_{mean}^{e}$"),
    ("QSenSim_mean_sbert",  "D", r"QS$_{mean}$"),

    ("QArgSim_max_exact_sbert", "E",r"QA$_{max}^{e}$"),
    ("QArgSim_max_sbert", "F", r"QA$_{max}$"),
    ("QArgSim_mean_exact_sbert", "G", "QA$_{mean}^{e}$"),
    ("QArgSim_mean_sbert", "H", r"QA$_{mean}$"),
]

# do renaming index
dict_for_rename_index = {}
index_order = []
for x in index_names:
    name = x
    if isinstance(x,tuple):
        name = x[1]
        dict_for_rename_index.update({x[0]:x[1]})
    index_order.append(name)

dict_for_rename_columns= {}
column_order = []
column_display = ["Participant"]
for x in column_names:
    name = x
    name_display = x
    if isinstance(x,tuple):
        name = x[1]
        name_display = x[2]
        dict_for_rename_columns.update({x[0]:x[1]})
    column_order.append(name)
    column_display.append(name_display)

combined_df.rename(index=dict_for_rename_index, columns=dict_for_rename_columns, inplace=True)

df_orderd = combined_df.reindex(columns=column_order, index=index_order)


#index=index_order
mewo = 1

# Convert rows to tuples
rows_as_tuples = [tuple(x) for x in df_orderd.to_records()]

x = tabulate(rows_as_tuples, tablefmt="latex_raw", headers=column_display)
print(x)


