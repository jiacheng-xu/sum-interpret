# Import pandas
import pandas as pd

# Assign spreadsheet filename to `file`
file = 'output.xlsx'

# Load spreadsheet
xl = pd.ExcelFile(file)

# Print the sheet names
print(xl.sheet_names)
