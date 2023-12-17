# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0.

from awscrt import mqtt, http
from awsiot import mqtt_connection_builder
from uuid import uuid4
import sys
import threading
import time
import json
import os
import requests
if __name__ == '__main__':
    # parse and load command-line parameter values

# create and format values for HTTPS request
    publish_url = 'https://' + os.environ["AWS_IOT_ENDPOINT"] + ':8443/topics/' + os.environ["TOPIC"] + '?qos=1'
    publish_msg = "hello world message"

    # make request
    publish = requests.request('POST',
            publish_url,
            data=publish_msg,
            cert=[os.environ["AWS_IOT_CERT"], os.environ["AWS_IOT_KEY"]])
    # # Create the proxy options if the data is present in cmdData
    # # proxy_options = None
    # # if cmdData.input_proxy_host is not None and cmdData.input_proxy_port != 0:
    # #     proxy_options = http.HttpProxyOptions(
    # #         host_name=cmdData.input_proxy_host,
    # #         port=cmdData.input_proxy_port)

    # # Create a MQTT connection from the command line data
    # print(os.environ["AWS_IOT_ENDPOINT"], os.environ["AWS_IOT_KEY"])
    # mqtt_connection = mqtt_connection_builder.mtls_from_path(
    #     endpoint=os.environ["AWS_IOT_ENDPOINT"],
    #     cert_filepath=os.environ["AWS_IOT_CERT"],
    #     pri_key_filepath=os.environ["AWS_IOT_KEY"],
    #     ca_filepath=os.environ["AWS_IOT_CA_FILE"],
    #     client_id="test-" + str(uuid4()))
    # connect_future = mqtt_connection.connect()

    # # Future.result() waits until a result is available
    # connect_future.result()
    # print("Connected!")

    # message_topic = os.environ["TOPIC"]
    # message_string = "message!"

    # # Subscribe
    # print("Subscribing to topic '{}'...".format(message_topic))
    # subscribe_future, packet_id = mqtt_connection.subscribe(
    #     topic=message_topic,
    #     qos=mqtt.QoS.AT_LEAST_ONCE)

    # subscribe_result = subscribe_future.result()
    # print("Subscribed with {}".format(str(subscribe_result['qos'])))

    # # Publish message to server desired number of times.
    # # This step is skipped if message is blank.
    # # This step loops forever if count was set to 0.
    # # if message_string:
    # #     if message_count == 0:
    # #         print("Sending messages until program killed")
    # #     else:
    # #         print("Sending {} message(s)".format(message_count))

    # message = "message: {}".format(message_string)
    # print("Publishing message to topic '{}': {}".format(message_topic, message))
    # message_json = json.dumps(message)
    # mqtt_connection.publish(
    #     topic=message_topic,
    #     payload=message_json,
    #     qos=mqtt.QoS.AT_LEAST_ONCE)
    # time.sleep(1)

    # # # Wait for all messages to be received.
    # # # This waits forever if count was set to 0.
    # # if message_count != 0 and not received_all_event.is_set():
    # #     print("Waiting for all messages to be received...")

    # # received_all_event.wait()
    # # print("{} message(s) received.".format(received_count))

    # # # Disconnect
    # # print("Disconnecting...")
    # # disconnect_future = mqtt_connection.disconnect()
    # # disconnect_future.result()
    # # print("Disconnected!")
