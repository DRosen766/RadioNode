from http import client
import urllib.parse
import json
import os
ip_addr = "localhost"
params = urllib.parse.urlencode({"word" : [1,2,3]})
print(params)
foo = {'word': 'Hello HTTP #1 **cool**, and #1!'}
json_data = json.dumps(foo)
radioConnection = client.HTTPConnection("{}:5000".format(ip_addr))
radioConnection.request("PUT", "/?{}".format(params), body="asdf")
