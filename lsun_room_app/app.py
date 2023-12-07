import asyncio
import multiprocessing as mp
import os
from pathlib import Path
from typing import Optional

from aiohttp import web
from aqueduct.integrations.aiohttp import (
    FLOW_NAME,
    AppIntegrator,
)
from typer import Typer

from .flow import (
    Flow,
    Task,
    get_flow,
)
from .utils import get_classes
from .version import __version__

typer_app = Typer(add_completion=False)
DEFAULT_PORT = 8014


class LsunRoomView(web.View):
    @property
    def flow(self: 'LsunRoomView') -> Flow:
        return self.request.app[FLOW_NAME]

    async def post(self: 'LsunRoomView') -> web.Response:
        post = await self.request.post()
        image = post.get('image')
        im = image.file.read()
        with_blend = bool(self.request.headers.get('with_blend', False))
        task = Task(image=im, with_blend=with_blend)
        await self.flow.process(task, timeout_sec=60)
        return web.Response(body=task.pred, content_type='image/png')


class LayoutClassesView(web.View):
    async def get(self: 'LayoutClassesView') -> web.Response:
        classes = get_classes()
        return web.json_response(classes)


async def get_version(request: web.Request) -> web.Response:
    return web.Response(text=__version__)


def prepare_app(weights_path: Path) -> web.Application:
    app = web.Application(client_max_size=0)
    app.router.add_post('/inference', LsunRoomView)
    app.router.add_get('/get_classes', LayoutClassesView)
    app.router.add_get('/version', get_version)

    AppIntegrator(app).add_flow(get_flow(weights_path))

    return app


@typer_app.command()
def main(
        weights_path: Optional[Path] = None,
        port: Optional[int] = None
) -> None:
    port = port or os.getenv('LSUN_ROOM_PORT') or DEFAULT_PORT
    loop = asyncio.get_event_loop()
    web.run_app(prepare_app(weights_path), loop=loop, port=port)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    typer_app()
