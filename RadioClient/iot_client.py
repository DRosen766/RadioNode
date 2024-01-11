# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0.

from awscrt import mqtt, http
from awsiot import mqtt_connection_builder
from uuid import uuid4
import time
import json
import os
import numpy as np
if __name__ == '__main__':
# create and format values for HTTPS request
    topic = os.environ["TOPIC"]
    mqtt_connection = mqtt_connection_builder.mtls_from_path(
        endpoint=os.environ['AWS_IOT_ENDPOINT'],
        cert_filepath=f"certs/{os.environ['AWS_IOT_CERT']}",
        pri_key_filepath=f"certs/{os.environ['AWS_IOT_KEY']}",
        ca_filepath=f"certs/{os.environ['AWS_IOT_CA_FILE']}",
        client_id="danny_rosen")
    connect_future = mqtt_connection.connect()

    # Future.result() waits until a result is available
    connect_future.result()
    print("Connected!")

    message_topic = topic
    message_string = "message!"

    # Subscribe
    print("Subscribing to topic '{}'...".format(message_topic))
    subscribe_future, packet_id = mqtt_connection.subscribe(
        topic=message_topic,
        qos=mqtt.QoS.AT_LEAST_ONCE)

    subscribe_result = subscribe_future.result()
    print("Subscribed with {}".format(str(subscribe_result['qos'])))

    # Publish message to server desired number of times.
    # This step is skipped if message is blank.
    # This step loops forever if count was set to 0.
    # if message_string:
    #     if message_count == 0:
    #         print("Sending messages until program killed")
    #     else:
    #         print("Sending {} message(s)".format(message_count))

    message = {"iq_data": list(np.ones(1024))}
    print("Publishing message to topic '{}': {}".format(message_topic, message))
    message_json = json.dumps(message)
    mqtt_connection.publish(
        topic=message_topic,
        payload=message_json,
        qos=mqtt.QoS.AT_LEAST_ONCE)
    time.sleep(1)

    # # Wait for all messages to be received.
    # # This waits forever if count was set to 0.
    # if message_count != 0 and not received_all_event.is_set():
    #     print("Waiting for all messages to be received...")

    # received_all_event.wait()
    # print("{} message(s) received.".format(received_count))

    # # Disconnect
    # print("Disconnecting...")
    # disconnect_future = mqtt_connection.disconnect()
    # disconnect_future.result()
    # print("Disconnected!")
