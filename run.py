from python_agent.agent import Agent, ReasoningStrategy
from python_agent.reasoning.plain import PlainReasoning
from python_agent.reasoning.chain_of_thought.chain_of_thought import ChainOfThought
from python_agent.reasoning.reasoning_base import ReasoningBase
from python_agent.tools.QueryKnowledgeGraphTool import QueryKnowledgeGraphTool
from python_agent.tools.QueryTimeSeriesDataPointsTool import QueryTimeSeriesDataPointsTool

import datetime
import json
from auth import create_client

cognite_client = create_client()


find_assets_tool = QueryKnowledgeGraphTool(cognite_client, {
    "space": "cdf_cdm",
    "version": "v1",
    "externalId": "CogniteCore"
}, ["CogniteAsset"], "agent_log.log")

find_maintenance_orders_tool = QueryKnowledgeGraphTool(cognite_client, {
    "space": "cdf_idm",
    "version": "v1",
    "externalId": "CogniteProcessIndustries"
}, ["CogniteMaintenanceOrder"], "agent_log.log")

find_time_series_tool = QueryKnowledgeGraphTool(cognite_client, {
    "space": "cdf_cdm",
    "version": "v1",
    "externalId": "CogniteCore"
}, ["CogniteTimeSeries"], "agent_log.log")

query_time_series_data_points_tool = QueryTimeSeriesDataPointsTool(cognite_client, "agent_log.log")

# Define a wrapper function that calls your tool.execute method.
def find_assets(query: str, operation: str):
    """Use this tool to find assets. Queries are typically on the format "search for asset X" or "list assets" or "list assets with parent with external id A and space B. Operation can be list, search or aggregate"""
    return find_assets_tool.execute(query+" Operation: "+operation)

def find_maintenance_orders(query: str, operation: str):
    """Use this tool to find maintenance orders.  Operation can be list, search or aggregate"""
    return find_maintenance_orders_tool.execute(query+" Operation: "+operation)

def find_time_series(query: str):
    """Use this tool to find time series. Queries are typically on the format "search for time series X" or "list time series" or "list time series for asset with external id A and space B """
    return find_time_series_tool.execute(query)

def query_time_series_data_points(space: str, externalId: str, start_iso8601: str, end_iso8601: str, num_data_points: int = 10):
    # Parse timestamps and ensure they have timezone info
    start = datetime.datetime.fromisoformat(start_iso8601)
    if start.tzinfo is None:
        start = start.replace(tzinfo=datetime.timezone.utc)
        
    end = datetime.datetime.fromisoformat(end_iso8601)
    if end.tzinfo is None:
        end = end.replace(tzinfo=datetime.timezone.utc)
    
    # Ensure start is at least 30 minutes before end
    if end - start < datetime.timedelta(minutes=30):
        start = end - datetime.timedelta(minutes=30)
    
    return query_time_series_data_points_tool.execute(space, externalId, start, end, num_data_points)



agent = Agent( 
    model="azure/gpt-4o-mini",
    reasoning_strategy=ReasoningStrategy.COT,
    debug=False,
    tools=[
        {'instance': find_assets_tool, 'function': find_assets},
        {'instance': find_time_series_tool, 'function': find_time_series},
        {'instance': query_time_series_data_points_tool, 'function': query_time_series_data_points},
        {'instance': find_maintenance_orders_tool, 'function': find_maintenance_orders}
    ]
)

import sys

if len(sys.argv) > 1:
    # Get input from command line argument
    user_input = " ".join(sys.argv[1:])
    response = agent.chat(user_input)
    print(f"Agent: {response}")
else:
    # Interactive mode
    print("Welcome to the terminal chat! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = agent.chat(user_input)
        print(f"Agent: {response}\n")