import sys

# get arguments from command line and set GPT_MODEL
GPT_MODEL = 'gpt-3.5-turbo' if len(sys.argv) > 1 else 'gpt-4'
