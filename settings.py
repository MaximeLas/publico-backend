import sys

# check if sqlite3 version is too old and install newer version if necessary in order to avoid error:
# Error: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0
import sqlite3
import subprocess
if sqlite3.sqlite_version_info < (3, 35, 0):
    print(f'sqlite3 version is too old version={sqlite3.sqlite_version_info}')
    print(f'installing sqlite3 version >= 3.35.0\n')

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pysqlite3-binary"])

    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

    print(f'\ninstalled sqlite3 version={sys.modules["sqlite3"].sqlite_version_info}\n')

import importlib.metadata
if importlib.metadata.version('gradio') < '3.35.0':
    print(f'gradio version is too old to use ClearButton version={importlib.metadata.version("gradio")}')
    # installing gradio version = 3.35.0 in order to use ClearButton
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio==3.35.0"])
    print(f'installed gradio version={importlib.metadata.version("gradio")}\n')

# get arguments from command line and set GPT_MODEL
GPT_MODEL = 'gpt-3.5-turbo' if len(sys.argv) > 1 else 'gpt-4'
