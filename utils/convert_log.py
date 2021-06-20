import argparse
import pandas as pd
# > define argument parser for file conversion
parser = argparse.ArgumentParser(description='convert specificed log metrics')
parser.add_argument('--filename', help='input filename')
args = parser.parse_args()

# > read in .xlsx file downloaded from Azure Metrics
excel_file = pd.read_excel(args.filename)[10:]

# > rename columns
excel_file.columns = ['Time','Total']
excel_file['Total'] = excel_file['Total'].astype('int64')

# > verify data format matches input specifications
print(excel_file[:3])
print(excel_file[-3:])

# > write to .csv stored in working directory
excel_file.to_csv(args.filename.replace('.xlsx','.csv'))
