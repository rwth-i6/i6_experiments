import os
import textwrap
from typing import Iterator

from sisyphus import Job, Task, Path


class WritePowerConsumptionScriptJob(Job):
    def __init__(self, binary_path: Path):
        super().__init__()

        self.binary_path = binary_path

        self.out_script = self.output_path("measure-power.py")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        script = textwrap.dedent(
            f"""
            #!/usr/bin/env python3

            import json
            import subprocess
            import sys
            import time

            import paho.mqtt.client as paho

            client = paho.Client(client_id="", userdata=None, protocol=paho.MQTTv5)

            class CallBackManager():
                def __init__(self, ):
                    self.time = None
                    self.ws = 0

                def on_connect(self, client, userdata, flags, rc):
                    print("Connected with result code "+str(rc))

                    # Subscribing in on_connect() means that if we lose the connection and
                    # reconnect then subscriptions will be renewed.
                    client.subscribe("gude1/#")
                    self.time = time.time()

                # The callback for when a PUBLISH message is received from the server.
                def on_message(self, client, userdata, msg):
                    print(msg.topic)
                    # print(msg.topic+" "+str(msg.payload))
                    d = json.loads(msg.payload)
                    power = d["line_in"][0]["voltage"] * d["line_in"][0]["current"]
                    print("current W:", power)
                    current_time = time.time()
                    self.ws += power * (current_time - self.time)
                    self.time = current_time

                    print("current Wh:", self.ws/3600)

            cbm = CallBackManager()
            client = paho.Client()
            client.on_connect = cbm.on_connect
            client.on_message = cbm.on_message

            client.username_pw_set("i6", password="1801")
            client.connect("137.226.223.2", 1883, 60)

            client.loop_start()
            try:
                subprocess.run([{self.binary_path}, *sys.argv[1:]])
            finally:
                client.disconnect()
            """
        )

        with open(self.out_script, "wt") as f:
            f.write(script)
        os.chmod(self.out_script, 0o755)
