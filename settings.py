import sys


# check if sqlite3 version is too old and install newer version if necessary in order to avoid error:
# Error: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0
'''import sqlite3
if sqlite3.sqlite_version_info < (3, 35, 0):
    print(f'sqlite3 version is too old version={sqlite3.sqlite_version_info}')
    print(f'installing sqlite3 version >= 3.35.0\n')
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pysqlite3-binary"])
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    print(f'\ninstalled sqlite3 version={sys.modules["sqlite3"].sqlite_version_info}\n')'''


# get arguments from command line and set GPT_MODEL to gpt-3.5-turbo if any argument is passed
GPT_MODEL = 'gpt-3.5-turbo' if len(sys.argv) > 1 else 'gpt-4'
