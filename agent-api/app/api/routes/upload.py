"""File upload endpoints"""

from fastapi import APIRouter, UploadFile, File
from typing import Dict, Any, List

router = APIRouter()

@router.post("/{session_id}")
async def upload_files(
    session_id: str, 
    files: List[UploadFile] = File(...)
) -> Dict[str, Any]:
    """Upload files for research session"""
    # TODO: Process uploaded files
    filenames = [file.filename for file in files]
    return {
        "session_id": session_id,
        "uploaded_files": filenames,
        "status": "uploaded"
    }
