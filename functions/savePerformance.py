import pandas as pd
import array

modeldata = pd.read_excel(r"C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\results\modelsdata.xlsx")
nums = list(range(1,len(modeldata)+1))
formattedList = ["Test%.0f" % n for n in nums]