from python_agent.tools.AgentTool import AgentTool
import datetime
import json

class QueryTimeSeriesDataPointsTool(AgentTool):
    def __init__(self, cognite_client, log_file_name: str):
        super().__init__(cognite_client, log_file_name)
        
    def calculate_granularity(self, start_ts: datetime.datetime, end_ts: datetime.datetime, num_points: int) -> str:
        import math
        # Calculate the total time difference in seconds between the two datetime objects
        time_difference_seconds = (end_ts - start_ts).total_seconds()
        
        # Compute target granularity in seconds, rounded up
        target_granularity_seconds = math.ceil(time_difference_seconds / num_points)
        
        # Convert to higher units
        target_granularity_minutes = math.ceil(target_granularity_seconds / 60)
        target_granularity_hours = math.ceil(target_granularity_minutes / 60)
        target_granularity_days = math.ceil(target_granularity_hours / 24)
        
        # Determine the appropriate granularity string
        if target_granularity_seconds <= 60:
            return f"{target_granularity_seconds}s"
        if target_granularity_minutes <= 60:
            return f"{target_granularity_minutes}m"
        if target_granularity_hours <= 24:
            return f"{target_granularity_hours}h"
        if target_granularity_days <= 100:
            return f"{target_granularity_days}d"
        
        return "100d"

    def execute(
        self,
        space: str,
        externalId: str,
        start: datetime.datetime,
        end: datetime.datetime,
        num_data_points: int = 10,
    ):
        with open(self.log_file_name, "a", encoding='utf-8') as f:
            f.write(f" [Thinking ...] Querying time series data points for {externalId} in space {space} from {start} to {end} with {num_data_points} data points\n")
        import cognite.client.data_classes.filters as flt
        # 1. Retrieve timeseries metadata using the instances endpoint
        ts_response = self.cognite_client.data_modeling.instances.list(
            sources=[
                {
                    "source": {
                        "externalId": "CogniteTimeSeries",
                        "space": "cdf_cdm",
                        "version": "v1",
                        "type": "view",
                    }
                }
            ],
            filter=flt.And(flt.Equals(['node', 'space'], value=space), flt.Equals(['node', 'externalId'], value=externalId)),
        )
        if not ts_response:
            return {
                "message": f"Unable to find timeseries: {externalId} in space: {space}"
            }
        timeseries = ts_response[0]

        # 2. Convert start and end times to UTC and then to UNIX timestamps (ms)
        start_utc = start.astimezone(datetime.timezone.utc)
        end_utc = end.astimezone(datetime.timezone.utc)
        start_ts = int(start_utc.timestamp() * 1000)
        end_ts = int(end_utc.timestamp() * 1000)

        # 3. Calculate granularity for roughly 10 data points
        granularity = self.calculate_granularity(start_utc, end_utc, num_data_points)

        # 4. Query aggregates using a POST request to the timeseries data list endpoint
        list_url = f"/api/v1/projects/{self.cognite_client.config.project}/timeseries/data/list"
        list_payload = {
            "items": [
                {
                    "instanceId": {"space": space, "externalId": externalId},
                    "start": start_ts,
                    "end": end_ts,
                    "aggregates": ["average", "min", "max"],
                    "granularity": granularity,
                }
            ]
        }
        list_response = self.cognite_client.post(
            list_url, headers={"cdf-version": "alpha"}, json=list_payload
        )
        points = list_response.json()['items'][0]["datapoints"]

        # 5. Retrieve the latest datapoint via the timeseries data latest endpoint
        latest_url = f"/api/v1/projects/{self.cognite_client.config.project}/timeseries/data/latest"
        latest_payload = {
            "items": [
                {
                    "instanceId": {"space": space, "externalId": externalId},
                }
            ]
        }
        latest_response = self.cognite_client.post(
            latest_url, headers={"cdf-version": "alpha"}, json=latest_payload
        )
        latest_items = latest_response.json()["items"]
        if latest_items and latest_items[0]["datapoints"]:
            latest_dp = latest_items[0]["datapoints"][0]
        else:
            latest_dp = {
                "timestamp": int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000),
                "value": None,
            }
        # 6. Convert the latest datapoint timestamp and aggregate points to local time strings
        latest_dp_local = {
            "timestamp": datetime.datetime.fromtimestamp(latest_dp["timestamp"] / 1000).isoformat(),
            "value": latest_dp["value"],
        }
        data_points_local = []
        for pt in points:
            pt_copy = pt.copy()
            pt_copy["timestamp"] = datetime.datetime.fromtimestamp(pt["timestamp"] / 1000).isoformat()
            data_points_local.append(pt_copy)

        # 7. If no aggregated points are returned, report that
        if not points:
            return {
                "message": "There are no datapoints in this timeseries, or in the given range"
            }
        self.current_thought_log.append(f"[Tool call: Query time series data points]:\n Time series: {externalId}\n Space: {space}\n Start: {start}\n End: {end}\n Number of data points: {num_data_points}")

        # 8. Construct a message combining the latest datapoint and the aggregates
        result_message = (
            f"Here is the latest data point:\n{json.dumps(latest_dp_local, indent=2)}.\n\n"
            f"Here are {num_data_points} aggregates:\n{json.dumps(data_points_local, indent=2)}"
        )
        # 9. Return a response with a display configuration and the message
        return result_message

