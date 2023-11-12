from http import client
import urllib.parse
import json
import os
ip_addr = "localhost"
# params = urllib.parse.urlencode({"@name" : "sentname"})
# foo = {'name': 'Hello HTTP #1 **cool**, and #1!'}
# json_data = json.dumps(foo)
radioConnection = client.HTTPConnection("{}:5000".format(ip_addr))
radioConnection.request("GET", "/")
