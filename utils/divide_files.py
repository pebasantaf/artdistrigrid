import glob
import shutil
path = r'C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\Data02'
dest = r'C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\Test02'
test_files = glob.glob(path + "/*.csv")# keeping directories in a list
halftomove = test_files[1:-1:2]
for f in halftomove:
    shutil.move(f, dest)
