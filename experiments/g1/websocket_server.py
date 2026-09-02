"""OmniRobot-compatible websocket transport for the stateful G1 policy."""

from __future__ import annotations

import asyncio
import http
import logging
import traceback
from typing import Any

import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames

from experiments.g1 import msgpack_numpy

logger = logging.getLogger(__name__)


class G1WebsocketPolicyServer:
    """Serve one stateful G1 policy using OmniRobot's native wire protocol.

    Metadata is sent immediately after connection. Each subsequent binary frame
    is one MessagePack observation and receives one MessagePack action reply.
    Only one client is allowed because RollingWAM carries a mutable rolling window.
    """

    def __init__(
        self,
        policy: Any,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = str(host)
        self._port = int(port)
        self._metadata = metadata or {}
        self._client_active = False
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with websocket_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info("G1 policy server listening on ws://%s:%d", self._host, self._port)
            await server.serve_forever()

    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        if self._client_active:
            message = "Another G1 policy client is already active."
            await websocket.send(message)
            await websocket.close(code=1013, reason=message)
            return

        self._client_active = True
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()

        try:
            self._policy.reset()
            await websocket.send(packer.pack(self._metadata))
            while True:
                raw = await websocket.recv()
                if isinstance(raw, str):
                    raise TypeError("G1 inference requests must be binary MessagePack frames.")
                observation = msgpack_numpy.unpackb(raw)
                if not isinstance(observation, dict):
                    raise TypeError(
                        f"Decoded observation must be a dict, got {type(observation).__name__}."
                    )
                action = self._policy.infer(observation)
                await websocket.send(packer.pack(action))
        except websockets.ConnectionClosed:
            logger.info("Connection from %s closed", websocket.remote_address)
        except Exception:
            error = traceback.format_exc()
            logger.error("Error during G1 inference:\n%s", error)
            try:
                await websocket.send(error)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
            except websockets.ConnectionClosed:
                pass
            raise
        finally:
            try:
                self._policy.reset()
            except Exception:
                logger.exception("Failed to reset G1 policy after client disconnect")
            self._client_active = False


def _health_check(
    connection: websocket_server.ServerConnection,
    request: websocket_server.Request,
) -> websocket_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
