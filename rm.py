import os
import shutil

for dirPath, dirNames, fileNames in os.walk("./save_"):
    if len(dirNames) < 2 and len(fileNames) < 3:
        try:
            shutil.rmtree(dirPath)
            print('delete ' + dirPath)
        except:
            print('Failed on delete ' + dirPath)