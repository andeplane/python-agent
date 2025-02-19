from python_agent.tools.AgentTool import AgentTool
from python_agent.tools.LLMTool import LLMTool
from dataclasses import dataclass, field
import json
import os
from typing import Any, Dict, List


@dataclass
class QueryKnowledgeGraphTool(AgentTool):
    data_model: Dict[str, Any] = field(init=False)
    views: List[str] = field(init=False)
    
    def __post_init__(self):
        # Call the base class post-init to initialize logs
        super().__post_init__()
        self.data_model = {
            "space": self.configuration['dataModels'][0]['space'],
            "version": self.configuration['dataModels'][0]['version'],
            "externalId": self.configuration['dataModels'][0]['externalId'],
        }
        self.views = self.configuration['dataModels'][0]['views']
        with open(os.path.join(os.path.dirname(__file__), "system_prompt.txt"), "r") as f:
            self.system_prompt_contribution = f.read()
        
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
                f.write(f"[Thinking ...]\nQuery Knowledge Graph query: {query}.\n")
                if query.get("operation") == "aggregate":
                    f.write(f"[Thinking ...]\nQuery Knowledge Graph result: {items}.\n")
                else:
                    #f.write(f"[Thinking ...]\nQuery Knowledge Graph result: {len(items)} items.\n")
                    f.write(f"[Thinking ...]\nQuery Knowledge Graph result: {json.dumps(items, indent=1)}.\n")

            # Add to the internal thought log
            self.current_thought_log.append(
                f"[Tool call: Query knowledge graph]:\n Question: {prompt}\n Views: {self.views}\n Generated query: {json.dumps(query)}\n Number of instances returned: {len(items)}"
            )

            for item in items:
                item["Instance type"] = query["dataModelView"]

            return (
                f"I generated the following query: {query}\n"
                f"Which gave the following instances from CDF: {items}\n"
                f"{aggregation_results if aggregation_results else ''}"
            )
        except Exception as e:
            with open(self.log_file_name, "a", encoding="utf-8") as f:
                f.write("[Thinking ...]\nQuery Knowledge Graph Error: " + str(e) + "\n\n")
            raise e
    
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