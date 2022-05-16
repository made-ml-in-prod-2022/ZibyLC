import pandas as pd
import pandas_profiling

df = pd.read_csv('./data/raw/heart_cleveland_upload.csv')

report = df.profile_report(title='pandas_profiling report', progress_bar=False)
report.to_file("reports/eda_report.html")
report.to_file("reports/eda_report.json")