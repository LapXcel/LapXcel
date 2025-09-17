import asyncio
import json
import re

def replace_float_notation(string):
    """
    Replace unity float notation for languages like
    French or German that use comma instead of dot.
    This convert the json sent by Unity to a valid one.
    Ex: "test": 1,2, "key": 2 -> "test": 1.2, "key": 2
    """
    regex_french_notation = r'"[a-zA-Z_]+":(?P<num>[0-9,E-]+),'
    regex_end = r'"[a-zA-Z_]+":(?P<num>[0-9,E-]+)}'
    for regex in [regex_french_notation, regex_end]:
        matches = re.finditer(regex, string, re.MULTILINE)
        for match in matches:
            num = match.group('num').replace(',', '.')
            string = string.replace(match.group('num'), num)
    return string

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
    """Handles messages from a single TCP client."""
    def __init__(self, msg_handler=None, chunk_size=16*1024):
        self.msg_handler = msg_handler
        self.chunk_size = chunk_size
        self.transport = None
        self.data_to_read = []
        self.data_to_write = asyncio.Queue()

    def connection_made(self, transport):
        print("DEBUG: SimHandler connection made")
        self.transport = transport
        if self.msg_handler:
            print("DEBUG: Calling msghandler.onconnect")
            self.msg_handler.on_connect(self)
        # Start writer task
        asyncio.create_task(self._writer())

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
            chunk = replace_float_notation(chunk)
            json_obj = json.loads(chunk)
        except Exception as e:
            print(e, 'failed to read json', chunk)
            return
        try:
            if self.msg_handler:
                self.msg_handler.on_recv_message(json_obj)
        except Exception as e:
            print(e, '>>> failure during on_recv_message:', chunk)

    def queue_message(self, msg):
        json_msg = json.dumps(msg)
        self.data_to_write.put_nowait(json_msg.encode())

    async def _writer(self):
        while True:
            data = await self.data_to_write.get()
            if data is None:
                break
            self.transport.write(data)

    def connection_lost(self, exc):
        if self.msg_handler:
            self.msg_handler.on_disconnect()
        print('Connection dropped')
        if self.msg_handler:
            self.msg_handler.on_close()

class SimServer:
    """Receives network connections and establishes handlers for each client."""
    def __init__(self, address, msg_handler):
        self.address = address
        self.msg_handler = msg_handler
        self.server = None

    async def start(self):
        print(f"Starting server on {self.address}")
        loop = asyncio.get_running_loop()
        self.server = await loop.create_server(
            lambda: SimHandler(msg_handler=self.msg_handler),
            self.address[0],
            self.address[1]
        )
        print(f'Binding to {self.address}')
        async with self.server:
            await self.server.serve_forever()
        print("Server created and listening")

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("Server shutdown")
            if self.msg_handler:
                self.msg_handler.on_close()
