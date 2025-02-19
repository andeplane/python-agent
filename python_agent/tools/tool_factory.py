from typing import Any, Dict, Type
from python_agent.tools.AgentTool import AgentTool
from python_agent.tools.QueryKnowledgeGraphTool.QueryKnowledgeGraphTool import QueryKnowledgeGraphTool

TOOL_TYPE_MAPPING: Dict[str, Type[AgentTool]] = {
    "queryDataModel": QueryKnowledgeGraphTool
}

def create_agent_tool(tool_data: Dict[str, Any], log_file_name: str) -> AgentTool | None:
    """
    Factory function to create an AgentTool subclass instance based on the 'type' field.
    
    Args:
        tool_data (Dict[str, Any]): Dictionary with tool configuration (typically loaded from JSON).
    
    Returns:
        AgentTool: An instance of the appropriate subclass.
    
    Raises:
        ValueError: If the tool type is not recognized.
    """
    tool_type = tool_data.get('type')
    assert isinstance(tool_type, str)

    tool_cls = TOOL_TYPE_MAPPING.get(tool_type)
    if not tool_cls:
        print(f"Ignoring unsupported tool type: {tool_type}")
        return None
        #raise ValueError(f"Unsupported tool type: {tool_type}")
    
    return tool_cls(**tool_data, log_file_name=log_file_name)