from auth import create_client
from typing import Any

def send_cog_ai_request(
    url: str,
    method: str,
    payload: dict[str, Any] | None = None,
    is_alpha: bool = True,
):
    client = create_client()
    request_kwargs = {}
    if payload:
        request_kwargs["json"] = payload
    if is_alpha:
        request_kwargs["headers"] = {"cdf-version": "alpha"}
    request_kwargs["url"] = url
    if method == "POST":
        method_fn = client.post
    elif method == "GET":
        method_fn = client.get
    return method_fn(**request_kwargs)