"""Job endpoints — start a file transcription, list jobs, cancel or dismiss one."""

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.core.constants import MAX_UPLOAD_SIZE
from app.pipeline.jobs import JobQueue, JobView, RemovalOutcome
from app.pipeline.upload_validation import read_upload_with_limit, validate_audio_upload

router = APIRouter()


class JobCreated(BaseModel):
    id: str


class JobRemoved(BaseModel):
    outcome: RemovalOutcome


def get_job_queue(request: Request) -> JobQueue:
    """FastAPI dependency — the app-lifetime ``JobQueue`` built in lifespan startup."""
    return request.app.state.jobs


@router.post("/file", response_model=JobCreated)
async def start_file_job(file: UploadFile, queue: JobQueue = Depends(get_job_queue)):
    """Check the upload's extension, content and size, queue it, and answer its id at once."""
    content = await read_upload_with_limit(file, MAX_UPLOAD_SIZE)
    validate_audio_upload(content, file.filename)
    return JobCreated(id=queue.add_file(content, file.filename or ""))


@router.get("", response_model=list[JobView])
async def list_jobs(queue: JobQueue = Depends(get_job_queue)):
    return queue.views()


@router.delete("/{job_id}", response_model=JobRemoved)
async def cancel_or_dismiss_job(job_id: str, queue: JobQueue = Depends(get_job_queue)):
    outcome = queue.cancel_or_dismiss(job_id)
    if outcome is None:
        raise HTTPException(status_code=404, detail="No such job")
    return JobRemoved(outcome=outcome)
