import pandas as pd
from influxdb_client import InfluxDBClient, Point, WriteOptions
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables

# Configuration
INFLUXDB_URL = os.getenv("INFLUXDB_URL")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET")

print(INFLUXDB_URL)


def EpochData(learningRate, error):
    return {"learning_rate": learningRate, "error": error}


def get_epoch_data(filename, epoch):
    def _get_last_epoch(epoch_data):
        return max([ep for ep in epoch_data.keys() if epoch_data[ep]["error"]])

    if not os.path.exists(filename):
        return None
    if os.path.isdir(filename):
        if os.path.exists(os.path.join(filename, "work/learning_rates")):
            filename = os.path.join(filename, "work/learning_rates")
        else:
            return None
    with open(filename, "r") as f:
        data = eval(f.read())
    if epoch is None:
        return data
    last_epoch = _get_last_epoch(data)  # sorted(list(data.keys()))[-1]
    if epoch == "last":
        epoch = last_epoch
    else:
        try:
            epoch = int(epoch)
        except ValueError:
            raise ValueError('epoch must be "last" or int')
        epoch = min(epoch, last_epoch)
    progr = data[epoch]
    progr["epoch"] = epoch
    return progr


from datetime import datetime


def fix_time(timestr: str):
    dt = datetime.strptime(timestr, "%Y-%m-%d-%H-%M-%S (UTC%z)")

    return dt.isoformat()


# Upload Data to InfluxDB
def upload_to_influxdb(data, name):
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    print(client.health())
    with client.write_api(write_options=WriteOptions(batch_size=500)) as write_api:
        for epoch, value in data.items():
            value: EpochData
            error_dict = value["error"]
            point = (
                Point("training_metrics").tag("name", name)
                # .field("learning_rate", value["learning_rate"])
                .field("epoch", epoch)
            )
            point = point.time(fix_time(error_dict[":meta:time"]))
            for key, val in error_dict.items():
                if key == ":meta:time":
                    continue
                point = point.field(key, val)

            write_api.write(INFLUXDB_BUCKET, INFLUXDB_ORG, point)

    client.close()


# Main Script
if __name__ == "__main__":
    data = get_epoch_data("./learning_rates.data", None)
    # log_file = "/path/to/training_logs.csv"
    # data = parse_training_logs(log_file)
    upload_to_influxdb(data, "test")
