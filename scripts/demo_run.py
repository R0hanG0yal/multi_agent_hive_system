import json
from src.environment import run_query

if __name__ == '__main__':
    out = run_query('What is a healthy snack to eat while coding?')
    print(json.dumps(out, indent=2))
