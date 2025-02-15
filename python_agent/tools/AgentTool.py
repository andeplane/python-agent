from cognite.client import CogniteClient
class AgentTool:
    def __init__(self, cognite_client: CogniteClient, log_file_name: str):
        self.cognite_client = cognite_client
        self.log_file_name = log_file_name
        self.current_thought_log: list[str] = []
        self.all_thoughts_log: list[str] = []
    
    def reset_thought_log(self):
        self.all_thoughts_log.extend(self.current_thought_log)
        self.current_thought_log = []

    def retrieve_current_thoughts_log(self) -> str:
        return '\n'.join(self.current_thought_log)
    
    def system_prompt_contribution(self) -> str:
        return ""
    
    def execute(self, query: str) -> str:
        """
        Abstract execute function that should be implemented by subclasses.
        
        Args:
            query (str): The query to execute
            
        Raises:
            NotImplementedError: This is an abstract method that must be implemented
        """
        raise NotImplementedError("Subclasses must implement execute()")