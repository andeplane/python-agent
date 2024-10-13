from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
from cognite.client import CogniteClient
from cognite.client.data_classes.data_modeling.ids import NodeId, ViewId

import pandas as pd


class CDFService(Enum):
    """
    Enum representing different Cognite Data Fusion services.
    """
    DATA_MODELING = "data_modeling"
    TIME_SERIES = "time_series"
    FILES = "files"

@dataclass
class InstanceRef:
    """
    Reference to a specific instance within a space.
    """
    space: str
    external_id: str


@dataclass
class DataModelingInstanceReference:
    """
    Reference structure for Data Modeling instances.
    """
    view_ref: ViewId
    instance_ref: NodeId


@dataclass
class TimeSeriesReference:
    """
    Reference structure for Time Series data.
    """
    id: Optional[int] = None
    external_id: Optional[str] = None
    instance_ref: Optional[InstanceRef] = None


@dataclass
class FileReference:
    """
    Reference structure for Files.
    """
    id: Optional[int] = None
    external_id: Optional[str] = None
    instance_ref: Optional[InstanceRef] = None


@dataclass
class CDFReference:
    """
    General reference structure for any Cognite Data Fusion service.
    """
    cdf_service: CDFService
    data_modeling_reference: DataModelingInstanceReference | None
    time_series_reference: TimeSeriesReference | None
    file_reference: FileReference | None


@dataclass
class TabularData:
    """
    Structure to hold tabular data along with metadata.
    """
    total_count: int
    next_cursor: Optional[str]
    data: pd.DataFrame

    def fetch_data(self, limit: int, cdf_reference: CDFReference, client: CogniteClient) -> None:
        raise NotImplementedError(f"Fetching of data is not implemented yet")


@dataclass
class MemoryObject:
    """
    Structure to hold data in memory along with its description and reference.
    """
    data: Union[str, Dict[str, Any], TabularData]
    description: str
    cdf_reference: Optional[CDFReference] = None


class Memory:
    """
    A class to manage in-memory storage of various data types with references to Cognite Data Fusion.
    """
    def __init__(self, cognite_client: CogniteClient):
        """
        Initializes the Memory object with a CogniteClient.

        Args:
            cognite_client (CogniteClient): The Cognite client instance.
        """
        self.data: Dict[str, MemoryObject] = {}
        self.client = cognite_client

    def store_data(
        self,
        id: str,
        data: Union[str, Dict[str, Any], TabularData],
        description: str,
        cdf_reference: Optional[CDFReference] = None
    ) -> None:
        """
        Stores data in memory with an associated ID, description, and optional CDF reference.

        Args:
            id (str): Unique identifier for the data.
            data (Union[str, Dict[str, Any], TabularData]): The data to store.
            description (str): Description of the data.
            cdf_reference (Optional[CDFReference], optional): Reference to Cognite Data Fusion. Defaults to None.
        """
        self.data[id] = MemoryObject(data=data, description=description, cdf_reference=cdf_reference)

    def retrieve_data(self, id: str) -> MemoryObject:
        """
        Retrieves data from memory by its ID.

        Args:
            id (str): The unique identifier of the data.

        Returns:
            MemoryObject: The retrieved memory object.

        Raises:
            KeyError: If the ID does not exist in memory.
        """
        if id not in self.data:
            raise KeyError(f"Data with ID '{id}' not found in memory.")
        return self.data[id]

    def summary_for_context(self) -> str:
        """
        Generates a summary string of all stored data for context purposes.

        Returns:
            str: A summary of all memory objects.
        """
        summary_lines: list[str] = []
        for memory_id, memory in self.data.items():
            data_type = type(memory.data).__name__
            summary_line = f"ID: {memory_id}, Type: {data_type}, Description: {memory.description}"
            if memory.cdf_reference:
                summary_line += f", CDF Service: {memory.cdf_reference.cdf_service.value}"
            summary_lines.append(summary_line)
        summary_str = "\n".join(summary_lines)
        return summary_str

    def delete_data(self, id: str) -> None:
        """
        Deletes data from memory by its ID.

        Args:
            id (str): The unique identifier of the data to delete.

        Raises:
            KeyError: If the ID does not exist in memory.
        """
        if id not in self.data:
            raise KeyError(f"Cannot delete. Data with ID '{id}' not found in memory.")
        del self.data[id]

    def list_ids(self) -> list[str]:
        """
        Lists all stored data IDs.

        Returns:
            list: A list of all data IDs.
        """
        return list(self.data.keys())

    def update_description(self, id: str, new_description: str) -> None:
        """
        Updates the description of a stored memory object.

        Args:
            id (str): The unique identifier of the data.
            new_description (str): The new description to set.

        Raises:
            KeyError: If the ID does not exist in memory.
        """
        if id not in self.data:
            raise KeyError(f"Cannot update description. Data with ID '{id}' not found in memory.")
        self.data[id].description = new_description
