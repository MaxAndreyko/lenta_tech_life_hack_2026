from fastapi import APIRouter, UploadFile, File, Depends, Request

from backend.app.api.response import ProcessRequest

router = APIRouter()


def get_controller(request: Request):
    return request.app.state.video_controller


@router.post("/process")
async def process_video(
    request: Request,
    file: UploadFile = File(...),
    params: ProcessRequest = Depends()
):
    controller = get_controller(request)
    return await controller.process_video(file, params)


@router.get("/download/{video_id}")
async def download_csv(request: Request, video_id: str):
    controller = get_controller(request)
    return await controller.download_csv(video_id)


@router.delete("/results/{video_id}")
async def delete_results(request: Request, video_id: str):
    controller = get_controller(request)
    return await controller.delete_results(video_id)