import oci.config
import oci.object_storage
from fastapi import HTTPException

from app.config import OCI_BUCKET
from app.state import state


def get_oci_client():
    if state.oci_client is None:
        config              = oci.config.from_file()
        state.oci_client    = oci.object_storage.ObjectStorageClient(config)
        state.oci_namespace = state.oci_client.get_namespace().data
    return state.oci_client, state.oci_namespace


def fetch_from_oci(object_name: str) -> bytes:
    try:
        client, namespace = get_oci_client()
        response = client.get_object(namespace, OCI_BUCKET, object_name)
        return response.data.content
    except Exception as exc:
        raise HTTPException(404, detail=f"OCI fetch failed for '{object_name}': {exc}")