import seaborn as sns
import pandas as pd
import numpy as np
import os
import csv
f = "/mnt/data0/jcxu/sent_plot.csv"


with open(f, 'r') as fd:
    csv_reader = csv.reader(fd, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
            line_count += 1

sns.set_theme(style="whitegrid")


values = rs.randn(365, 4).cumsum(axis=0)
dates = pd.date_range("1 1 2016", periods=365, freq="D")
data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
data = data.rolling(7).mean()

sns.lineplot(data=data, palette="tab10", linewidth=2.5)

sns.lineplot()
