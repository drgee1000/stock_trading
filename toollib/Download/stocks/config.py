import subprocess
import shutil
import os
import zipline

def create_config():
    cwd = os.getcwd()
    with open(cwd+"/data/stocks/extension.py","r") as f:
        lines = f.readlines()

    with open(cwd+"/data/stocks/extension.py","r+") as f1:
        f1.seek(0)
        lines[11] = '\t\''+cwd+"/csv/stocks"+'\',\n'
        for line in lines:
            f1.write(line)
        f1.truncate()
    path = os.path.expanduser(zipline.utils.paths.zipline_root())
    if not os.path.isdir(path):
        os.mkdir(path)
    shutil.copyfile('data/stocks/extension.py', path+'/extension.py')
    subprocess.call(["zipline", "ingest", "-b", 'custom-stocks-csvdir-bundle'])
