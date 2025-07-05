"""
Socket Transport Module

This module implements a socket-based transport for MCP that provides
1-to-1 client-server communication over TCP sockets.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, TextIO

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import BaseModel, Field

import mcp.types as types
from mcp.shared.message import SessionMessage

logger = logging.getLogger(__name__)


class SocketServerParameters(BaseModel):
    """Configuration parameters for socket-based transport."""

    command: str
    """The executable to run to start the server."""

    args: list[str] = Field(default_factory=list)
    """Command line arguments to pass to the executable."""

    env: dict[str, str] | None = None
    """
    The environment to use when spawning the process.
    If not specified, the current environment will be used.
    """

    cwd: str | Path | None = None
    """The working directory to use when spawning the process."""

    host: str = Field(default="127.0.0.1")
    """The host to bind to for socket communication."""

    port: int = Field(default=0)
    """
    The port to bind to for socket communication.
    If 0, a random available port will be used.
    """

    encoding: str = "utf-8"
    """The text encoding used when sending/receiving messages."""

    encoding_error_handler: str = "strict"
    """The text encoding error handler."""

    connection_timeout: float = Field(default=5.0)
    """Timeout in seconds for connection acceptance."""


@asynccontextmanager
async def socket_client(
    server: SocketServerParameters, errlog: TextIO = sys.stderr
) -> AsyncGenerator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
    ],
    None,
]:
    """
    Client transport for socket-based communication.

    This will:
    1. Start a server process using the provided command
    2. Create a socket connection to that server
    3. Communicate using JSON-RPC messages over the socket connection

    Args:
        server: Socket server parameters
        errlog: Where to send server process stderr (defaults to sys.stderr)

    Yields:
        A tuple containing:
        - read_stream: Stream for reading messages from the server
        - write_stream: Stream for sending messages to the server

    Raises:
        TimeoutError: If connection acceptance times out
        OSError: If process startup fails
        Exception: For other errors
    """
    read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
    read_stream_writer: MemoryObjectSendStream[SessionMessage | Exception]

    write_stream: MemoryObjectSendStream[SessionMessage]
    write_stream_reader: MemoryObjectReceiveStream[SessionMessage]

    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    # Create a TCP listener first to get the port
    listener = await anyio.create_tcp_listener(
        local_host=server.host, local_port=server.port
    )
    actual_port = listener.extra(anyio.abc.SocketAttribute.local_port)
    logger.info(f"Listening on port {actual_port}")

    try:
        # Start the server process with the port as an argument
        process_args = [*server.args, "--port", str(actual_port)]
        process = await anyio.open_process(
            [server.command, *process_args],
            env=server.env or os.environ,
            stderr=errlog,
            cwd=server.cwd,
        )

        try:
            # Accept connection from the server with timeout
            stream = None
            connection_event = anyio.Event()
            shutdown_event = anyio.Event()

            async def handle_connection(client_stream):
                nonlocal stream
                stream = client_stream
                logger.info(f"Accepted connection from server")
                connection_event.set()

            async def run_listener():
                try:
                    async with listener:
                        await listener.serve(handle_connection)
                except anyio.get_cancelled_exc_class():
                    # Normal cancellation, just exit
                    logger.info("Listener cancelled")
                    pass
                except Exception as e:
                    logger.error(f"Error in listener: {e}")
                    raise

            async def socket_reader():
                """Reads messages from the socket and forwards them to read_stream."""
                try:
                    async with read_stream_writer:
                        buffer = ""
                        async for data in stream:
                            text = data.decode(
                                server.encoding, server.encoding_error_handler
                            )
                            lines = (buffer + text).split("\n")
                            buffer = lines.pop()

                            for line in lines:
                                try:
                                    message = types.JSONRPCMessage.model_validate_json(
                                        line
                                    )
                                    session_message = SessionMessage(message)
                                    await read_stream_writer.send(session_message)
                                except Exception as exc:
                                    logger.error(f"Error in socket reader: {exc}")
                                    await read_stream_writer.send(exc)
                                    continue
                except anyio.ClosedResourceError:
                    logger.info("Socket reader closed")
                    await anyio.lowlevel.checkpoint()
                    logger.info("Socket reader checkpointed")
                    # Signal that session is closing normally
                    shutdown_event.set()
                    return  # Exit normally
                except anyio.get_cancelled_exc_class():
                    logger.info("Socket reader cancelled")
                    return  # Exit normally on cancellation
                except Exception as e:
                    logger.error(f"Error in socket reader: {e}")
                    # Don't set shutdown_event on unexpected errors to avoid
                    # interfering with exception propagation
                    raise
                finally:
                    logger.info("=== Socket reader cleanup: closing stream ===")
                    await stream.aclose()
                    logger.info("=== Socket reader cleanup completed ===")

            async def socket_writer():
                """Reads messages from write_stream and sends them over the socket."""
                try:
                    async with write_stream_reader:
                        async for session_message in write_stream_reader:
                            json = session_message.message.model_dump_json(
                                by_alias=True, exclude_none=True
                            )
                            data = (json + "\n").encode(
                                server.encoding, server.encoding_error_handler
                            )
                            await stream.send(data)
                except anyio.ClosedResourceError:
                    logger.info("Socket writer closed")
                    await anyio.lowlevel.checkpoint()
                    logger.info("Socket writer checkpointed")
                    # Signal that session is closing normally
                    shutdown_event.set()
                    return  # Exit normally
                except anyio.get_cancelled_exc_class():
                    logger.info("Socket writer cancelled")
                    return  # Exit normally on cancellation
                except Exception as e:
                    logger.error(f"Error in socket writer: {e}")
                    # Don't set shutdown_event on unexpected errors to avoid
                    # interfering with exception propagation
                    raise
                finally:
                    logger.info("=== Socket writer cleanup: closing stream ===")
                    await stream.aclose()
                    logger.info("=== Socket writer cleanup completed ===")

            async def shutdown_monitor(tg):
                """Monitor for shutdown event and cancel task group when needed."""
                await shutdown_event.wait()
                logger.info("Shutdown event received, cancelling task group")
                # Give a small delay to let cleanup messages be logged
                await anyio.sleep(0.1)
                tg.cancel_scope.cancel()

            async with anyio.create_task_group() as tg:
                logger.info("=== Starting task group ===")
                # Start the listener task
                tg.start_soon(run_listener)

                # Start the shutdown monitor
                tg.start_soon(shutdown_monitor, tg)

                # Wait for connection with timeout
                with anyio.fail_after(server.connection_timeout):
                    await connection_event.wait()

                # Start reader and writer tasks
                tg.start_soon(socket_reader)
                tg.start_soon(socket_writer)

                try:
                    logger.info("Yielding streams to caller")
                    yield read_stream, write_stream
                except Exception as e:
                    # For any exception from the yield block (including test failures),
                    # cancel the task group immediately to prevent ExceptionGroup wrapping
                    logger.info(
                        f"Exception in yield block: {type(e).__name__}, cancelling task group"
                    )
                    tg.cancel_scope.cancel()
                    # Let tasks complete their cancellation
                    await anyio.sleep(0.01)
                    raise
                finally:
                    # Cancel all tasks and clean up
                    logger.info("=== Starting cleanup ===")
                    logger.info("Stage 1: Cancelling task group")
                    tg.cancel_scope.cancel()

                    logger.info("Stage 2: Terminating process")
                    # Clean up process to prevent any dangling orphaned processes
                    try:
                        if process.returncode is None:
                            logger.info("Process still running, terminating...")
                            process.terminate()
                        else:
                            logger.info(
                                f"Process already exited with code {process.returncode}"
                            )
                    except ProcessLookupError:
                        logger.info("Process already exited (ProcessLookupError)")
                    except Exception as e:
                        logger.warning(f"Error terminating process: {e}")

                    logger.info("Stage 3: Closing socket stream")
                    if stream:
                        try:
                            await stream.aclose()
                            logger.info("Socket stream closed")
                        except Exception as e:
                            logger.warning(f"Error closing socket stream: {e}")

                    logger.info("Stage 4: Closing streams")
                    await read_stream.aclose()
                    logger.info("- read_stream closed")
                    await write_stream.aclose()
                    logger.info("- write_stream closed")
                    await read_stream_writer.aclose()
                    logger.info("- read_stream_writer closed")
                    await write_stream_reader.aclose()
                    logger.info("- write_stream_reader closed")
                    logger.info("All streams closed successfully")
                    logger.info("=== Cleanup completed ===")

        finally:
            # Clean up process
            logger.info("=== Starting process cleanup ===")
            if process.returncode is None:
                logger.info(f"Terminating process {process.pid} in middle cleanup")
                process.terminate()
                logger.info("=== Process terminated ===")
            await process.aclose()
            logger.info("=== Process cleanup completed ===")

    finally:
        # Clean up listener
        logger.info("=== Starting listener cleanup ===")
        await listener.aclose()
        logger.info("=== Listener cleanup completed ===")
        logger.info("Socket client cleanup sequence completed")
