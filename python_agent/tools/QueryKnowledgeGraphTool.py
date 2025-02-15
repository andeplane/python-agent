from python_agent.tools.AgentTool import AgentTool
import json
class QueryKnowledgeGraphTool(AgentTool):
    def __init__(self, cognite_client, data_model, views, log_file_name: str):
        super().__init__(cognite_client, log_file_name)
        self.data_model = data_model
        self.views = views

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
            f.write(f" [Thinking ...] Query Knowledge Graph: {self.views[0]}: {prompt}\n")
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
            items = data.get("items", [])
            with open(self.log_file_name, "a", encoding='utf-8') as f:
                f.write(f" [Thinking ...] Query Knowledge Graph result: {len(items)} items.\n")
            
            self.current_thought_log.append(f"[Tool call: Query knowledge graph]:\n Question: {prompt}\n Views: {self.views}\n Generated query: {json.dumps(query)}\n Number of instances returned: {len(items)}")

            return f"""
                I generated the following query: {query}
                Which gave the following instances from CDF: {items}
            """
        except Exception as e:
            with open(self.log_file_name, "a", encoding='utf-8') as f:
                f.write("  [Thinking ...] Query Knowledge Graph Error: " + str(e) + "\n\n")
            return {"error": str(e)}
