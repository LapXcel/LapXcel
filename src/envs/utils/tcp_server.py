import asyncio
import json

class IMesgHandler:
    """Abstract class that represents a socket message handler."""
    def on_connect(self, socket_handler):
        pass
    def on_recv_message(self, message):
        pass
    def on_close(self):
        pass
    def on_disconnect(self):
        pass

class SimHandler(asyncio.Protocol):
    def __init__(self, msg_handler=None, chunk_size=16*1024, verbose=False):
        self.msg_handler = msg_handler
        self.chunk_size = chunk_size
        self.transport = None
        self.loop = None                   # store loop that runs this protocol
        self.data_to_read = []
        self.data_to_write = None          # create queue in loop (connection_made)
        self.verbose = verbose

    def connection_made(self, transport):
        if self.verbose:
            print("[SimHandler] connection made")
        self.transport = transport
        # store the event loop that is driving this protocol
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None
        # create queue on the loop thread
        if self.loop is not None:
            # must create the asyncio.Queue on the event loop thread
            # call_soon_threadsafe ensures it runs on the loop thread
            def _init_queue():
                self.data_to_write = asyncio.Queue()
                # start the writer task on the loop
                asyncio.create_task(self._writer())
            self.loop.call_soon_threadsafe(_init_queue)
        else:
            # fallback (shouldn't happen in normal asyncio server)
            self.data_to_write = asyncio.Queue()
            asyncio.create_task(self._writer())

        if self.msg_handler:
            if self.verbose:
                print("[SimHandler] Calling msghandler.onconnect")
            self.msg_handler.on_connect(self)

    def queue_message(self, msg):
        json_msg = json.dumps(msg) + "\n"
        if self.verbose:
            print("[Writer] Queuing message...", json_msg)

        # If we are on the same loop thread, put directly
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is not None and current_loop is self.loop:
            # on same event loop thread — safe
            if self.data_to_write is None:
                self.data_to_write = asyncio.Queue()
                asyncio.create_task(self._writer())
            self.data_to_write.put_nowait(json_msg.encode())
        else:
            # called from another thread — schedule the put on the server loop thread
            if self.loop is None:
                # no loop known — create queue safely on the caller thread (last resort)
                if self.data_to_write is None:
                    self.data_to_write = asyncio.Queue()
                    asyncio.create_task(self._writer())
                self.data_to_write.put_nowait(json_msg.encode())
            else:
                # thread-safe scheduling
                def _put():
                    if self.data_to_write is None:
                        self.data_to_write = asyncio.Queue()
                        asyncio.create_task(self._writer())
                    self.data_to_write.put_nowait(json_msg.encode())
                self.loop.call_soon_threadsafe(_put)


    def data_received(self, data):
        self.data_to_read.append(data.decode('utf-8'))
        messages = ''.join(self.data_to_read).split('\n')
        self.data_to_read = []
        for mesg in messages:
            if len(mesg) < 2:
                continue
            if mesg[0] == '{' and mesg[-1] == '}':
                self.handle_json_message(mesg)
            else:
                self.data_to_read.append(mesg)

    def handle_json_message(self, chunk):
        try:
            json_obj = json.loads(chunk)
        except Exception as e:
            print(e, 'failed to read json', chunk)
            return
        try:
            if self.msg_handler:
                self.msg_handler.on_recv_message(json_obj)
        except Exception as e:
            print(e, '>>> failure during on_recv_message')

    async def _writer(self):
        if self.verbose:
            print("[Writer] Initializing...")
        while True:
            data = await self.data_to_write.get()
            if data is None:
                break
            if self.verbose:
                print(f"[Writer] Writing data {data}...")
            self.transport.write(data)

    def connection_lost(self, exc):
        if self.msg_handler:
            self.msg_handler.on_disconnect()
        print('Connection dropped')
        if self.msg_handler:
            self.msg_handler.on_close()

class SimServer:
    """Receives network connections and establishes handlers for each client."""
    def __init__(self, address, msg_handler, verbose):
        self.address = address
        self.msg_handler = msg_handler
        self.server = None
        self.verbose = verbose

    async def start(self):
        if self.verbose:
            print(f"[SimServer] Starting server on {self.address}")
        loop = asyncio.get_running_loop()
        self.server = await loop.create_server(
            lambda: SimHandler(msg_handler=self.msg_handler),
            self.address[0],
            self.address[1]
        )
        if self.verbose:
            print(f'[SimServer] Binding to {self.address}')
        async with self.server:
            await self.server.serve_forever()
        if self.verbose:
            print("[SimServer] Server created and listening")

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("[SimServer] Server shutdown")
            if self.msg_handler:
                self.msg_handler.on_close()
