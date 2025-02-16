from python_agent.tools.AgentTool import AgentTool
import json
class QueryKnowledgeGraphTool(AgentTool):
    def __init__(self, cognite_client, data_model, views, log_file_name: str):
        super().__init__(cognite_client, log_file_name)
        self.data_model = data_model
        self.views = views

    def system_prompt_contribution(self) -> str:
        return f"""
        QueryKnowledgeGraphTool instructions:
        This tool allows query generation from natural language. The following things can be mentioned:
         - Operation. You can choose to list, search or aggregate.
         - Filters on properties. This typically is filters on datetime properties, but often also relations. 
           - In order to filter on relations, you always must provide the space and the externalId of the target node.
         - Sorting. You can sort on properties
         - Limit. You can limit the number of instances returned.
        Usually, data is centered around assets. Strategy is often to find the right asset, then perform list queries with filters on this asset instance (space+externalId).
        Alternatively, you can search directly on instances, but only when these instances are not coupled to the asset or this does not matter.
        Examples:
         - "Show me time series for asset in space <SPACE> with externalId <EXTERNALID>"
         - "Show me assets with parent space <SPACE> with externalId <EXTERNALID>"
         - "Show me activities sorted by start time.
         - "Search for assets with name <NAME>"
         - "How many activities were there in 2020?"
         - "Show me the 10 latest activities"
        """

    def execute_query(self, query):
        import json
        body = {
            "operation": query["operation"],
            "dataModelView": query["dataModelView"],
            "properties": query['properties'],
            "aggregate": query.get('aggregate', None),
            "search": query.get('search', None),
            "filter": query['filter'],
            "limit": query.get('limit', 20)
        }
        if "search" in query:
            if not "name" in query["search"]["fields"]:
                query["search"]["fields"].append("name")
        response = self.cognite_client.post(f'/api/v1/projects/{self.cognite_client.config.project}/ai/tools/query/execute',
                            headers={"cdf-version": "alpha"},
                            json=body)
        data = response.json()
        return data

    def execute(self, prompt: str):
        with open(self.log_file_name, "a", encoding='utf-8') as f:
            f.write(f"[Thinking ...]\nQuery Knowledge Graph: {self.views[0]}: {prompt}\n")
        body = {
            "prompt": prompt,
            "dataModels": [
                {
                    "views": self.views,
                    "space": self.data_model["space"],
                    "version": self.data_model["version"],
                    "externalId": self.data_model["externalId"]
                }
            ],
            "stream": False,
        }
        try:
            response = self.cognite_client.post(f'/api/v1/projects/{self.cognite_client.config.project}/ai/tools/query/generate',
                                headers={"cdf-version": "alpha"},
                                json=body)
            query = response.json()
            data = self.execute_query(query)
            aggregation_results = None
            if query['operation'] == 'list':
                aggregation_results = self.execute_query({**query, "operation": 'aggregate', "aggregate": {"properties": {"count": ["externalId"]}}})
            items = data.get("items", [])
            with open(self.log_file_name, "a", encoding='utf-8') as f:
                f.write(f"[Thinking ...]\nQuery Knowledge Graph result: {len(items)} items.\n")
            
            self.current_thought_log.append(f"[Tool call: Query knowledge graph]:\n Question: {prompt}\n Views: {self.views}\n Generated query: {json.dumps(query)}\n Number of instances returned: {len(items)}")

            return f"""
                I generated the following query: {query}
                Which gave the following instances from CDF: {items}
                {aggregation_results if aggregation_results else ""}
            """
        except Exception as e:
            with open(self.log_file_name, "a", encoding='utf-8') as f:
                f.write("[Thinking ...]\nQuery Knowledge Graph Error: " + str(e) + "\n\n")
            return {"error": str(e)}
