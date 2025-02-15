class AgentTool:
    def __init__(self, cognite_client, log_file_name: str):
        self.cognite_client = cognite_client
        self.log_file_name = log_file_name
        self.current_thought_log = []
        self.all_thoughts_log = []
    
    def reset_thought_log(self):
        self.all_thoughts_log.extend(self.current_thought_log)
        self.current_thought_log = []

    def retrieve_current_thoughts_log(self):
        return self.current_thought_log

    def execute(self, query: str):
        """
        Abstract execute function that should be implemented by subclasses.
        
        Args:
            query (str): The query to execute
            
        Raises:
            NotImplementedError: This is an abstract method that must be implemented
        """
        raise NotImplementedError("Subclasses must implement execute()")