from python_agent.tools.AgentTool import AgentTool
from python_agent.tools.LLMTool import LLMTool
from dataclasses import dataclass, field
from auth import create_client
import json
from typing import Any, Dict, List


@dataclass
class QueryKnowledgeGraphTool(AgentTool):
    data_model: Dict[str, Any] = field(init=False)
    views: List[str] = field(init=False)
    
    def __post_init__(self):
        # Call the base class post-init to initialize logs
        super().__post_init__()
        self.cognite_client = create_client()
        
        self.data_model = {
            "space": self.configuration['dataModels'][0]['space'],
            "version": self.configuration['dataModels'][0]['version'],
            "externalId": self.configuration['dataModels'][0]['externalId'],
        }
        self.views = self.configuration['dataModels'][0]['views']
        
        # self.system_prompt_contribution = (
        #     "QueryKnowledgeGraphTool instructions:\n"
        #     "This tool allows query generation from natural language. The following things can be mentioned:\n"
        #     " - Operation. You can choose to list, search or aggregate.\n"
        #     " - Filters on properties. This typically is filters on datetime properties, but often also relations.\n"
        #     "   - In order to filter on relations, you always must provide the space and the externalId of the target node.\n"
        #     " - Sorting. You can sort on properties.\n"
        #     " - Limit. You can limit the number of instances returned.\n"
        #     "Usually, data is centered around assets. The strategy is often to find the right asset, then perform list queries with filters on this asset instance (space+externalId).\n"
        #     "Alternatively, you can search directly on instances, but only when these instances are not coupled to the asset or it does not matter.\n"
        #     "Examples:\n"
        #     ' - "Show me time series for asset in space <SPACE> with externalId <EXTERNALID>"\n'
        #     ' - "Show me assets with parent space <SPACE> with externalId <EXTERNALID>"\n'
        #     ' - "Show me activities sorted by start time."\n'
        #     ' - "Search for assets with name <NAME>"\n'
        #     ' - "How many activities were there in 2020?"\n'
        #     ' - "Show me the 10 latest activities"\n'
        # )
        self.system_prompt_contribution = (
            "QueryKnowledgeGraphTool instructions:"
            "You will often find data, and the context will be Cognite's Data Model Service (DMS). \n"
            "To query data there, you can use tools that are specialized for that. \n"
            "General instructions on how to query data in DMS: \n"
            " - Specify that you want to search when a user specifies an identifier (like asset 18AB1234). This identifier often follows standards such as NORSOK. \n"
            " - If you want to find related data OtherType (e.g. work orders or files for asset X), find the asset first to get space and externalId. Then you can ask a question formulated like \"Find OtherType for asset with {space: <space>, externalId: <externalId>}\". \n"
            " - Try to be as explicit as you can when using these tools, but do not include too much information. If you want to perform search, specify that you want to perform search. \n"
            "Examples: \n"
            "### Find work orders for asset 18AB1234 \n"
            "Perform the following queries \n"
            " - Search for asset 18AB1234 \n"
            " - Find work orders for asset with {{space: \"<result from previous query>\", externalId: \"<result from previous query>\"}} \n"
            "### Find how many work orders for asset 13FV1820 this year \n"
            "Perform the following queries \n"
            " - Search for asset 13FV1820 \n"
            " - Count the number of work orders this year for asset with {{space: \"<result from previous query>\", externalId: \"<result from previous query>\"}} \n"
        )

    def execute_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        body = {
            "operation": query["operation"],
            "dataModelView": query["dataModelView"],
            "properties": query["properties"],
            "aggregate": query.get("aggregate"),
            "search": query.get("search"),
            "filter": query["filter"],
            "limit": query.get("limit", 20)
        }
        # Ensure that 'name' is included in the search fields if applicable
        if query.get("search") is not None:
            fields = query["search"].get("fields", [])
            if "name" not in fields:
                query["search"].setdefault("fields", []).append("name")
        response = self.cognite_client.post(
            f'/api/v1/projects/{self.cognite_client.config.project}/ai/tools/query/execute',
            headers={"cdf-version": "alpha"},
            json=body
        )
        return response.json()

    def execute(self, prompt: str) -> Any:
        # Log the prompt being executed
        with open(self.log_file_name, "a", encoding="utf-8") as f:
            f.write(f"[Thinking ...]\nQuery Knowledge Graph: {self.views[0]}: {prompt}\n")

        body = {
            "prompt": prompt,
            "dataModels": [
                {
                    "views": self.views,
                    "space": self.data_model["space"],
                    "version": self.data_model["version"],
                    "externalId": self.data_model["externalId"],
                }
            ],
            "stream": False,
        }

        try:
            # Generate the query based on the prompt
            response = self.cognite_client.post(
                f'/api/v1/projects/{self.cognite_client.config.project}/ai/tools/query/generate',
                headers={"cdf-version": "alpha"},
                json=body
            )
            query = response.json()
            data = self.execute_query(query)

            # Optionally, if the operation is "list", execute an aggregate query as well.
            aggregation_results = None
            if query.get("operation") == "list":
                aggregate_query = {
                    **query,
                    "operation": "aggregate",
                    "aggregate": {"properties": {"count": ["externalId"]}},
                }
                aggregation_results = self.execute_query(aggregate_query)

            items = data.get("items", [])
            with open(self.log_file_name, "a", encoding="utf-8") as f:
                f.write(f"[Thinking ...]\nQuery Knowledge Graph result: {len(items)} items.\n")

            # Add to the internal thought log
            self.current_thought_log.append(
                f"[Tool call: Query knowledge graph]:\n Question: {prompt}\n Views: {self.views}\n Generated query: {json.dumps(query)}\n Number of instances returned: {len(items)}"
            )

            return (
                f"I generated the following query: {query}\n"
                f"Which gave the following instances from CDF: {items}\n"
                f"{aggregation_results if aggregation_results else ''}"
            )
        except Exception as e:
            with open(self.log_file_name, "a", encoding="utf-8") as f:
                f.write("[Thinking ...]\nQuery Knowledge Graph Error: " + str(e) + "\n\n")
            return {"error": str(e)}
    
    def get_llm_tools(self) -> List[LLMTool]:
        """
        Returns a list containing a single LLMTool for this QueryKnowledgeGraphTool.
        """
        return [QueryKnowledgeGraphLLMTool(self)]
    
class QueryKnowledgeGraphLLMTool(LLMTool):
    def __init__(self, agent_tool: QueryKnowledgeGraphTool):
        self.agent_tool = agent_tool

    def invoke(self, **kwargs: Any) -> Any:
        """
        Invoke the QueryKnowledgeGraphTool with the given parameters.
        """
        prompt = kwargs['prompt']
        limit = kwargs.get('limit')
        operation = kwargs.get('operation')
        if operation:
            prompt += f" Operation: {operation}. "
        if limit:
            prompt += f" Limit: {limit}. "
        return self.agent_tool.execute(prompt)

    def get_json_schema(self) -> Dict[str, Any]:
        """
        Return the JSON schema for this tool.
        Schema:
            - prompt: string (required) - the natural language query.
            - limit: integer (optional) - number of instances to retrieve.
            - operation: string (optional) - the operation to perform. Can be 'list', 'search' or 'aggregate'.
        """
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Natural language query, be expressive.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of instances to retrieve.",
                    "default": 0
                },
                "operation": {
                    "type": "string",
                    "description": "Query operation to perform. Can be 'list', 'search' or 'aggregate'."
                }
            },
            "required": ["prompt"]
        }