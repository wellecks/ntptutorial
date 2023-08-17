import sys
import http.client
import json
import sys


def suggest(tactic_state):
    conn = http.client.HTTPConnection("localhost", 5000)
    headers = {'Content-type': 'application/json'}
    body = json.dumps({"tactic_state": sys.argv[1]})
    conn.request("POST", "/", body, headers)
    response = conn.getresponse()
    data = response.read()
    data_dict = json.loads(data)
    print('[SUGGESTION]'.join(data_dict['suggestions']))
    conn.close()

if __name__ == "__main__":
    suggest(sys.argv[1])