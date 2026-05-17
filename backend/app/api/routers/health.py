from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/")
async def root(request: Request):
    controller = request.app.state.health_controller
    return await controller.get_info()


@router.get("/health")
async def health(request: Request):
    controller = request.app.state.health_controller
    return await controller.get_health()