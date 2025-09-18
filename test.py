import socket
import json
import struct

def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 65431))
    sock.listen(0)
    sock.accept()

    # Send a 'request' message
    message = {"msg_type": "request"}
    msg_bytes = json.dumps(message).encode('utf-8')
    sock.sendall(msg_bytes)

    # Receive response size
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        print("No response length received")
        return
    msglen = struct.unpack('>I', raw_msglen)[0]

    # Receive response message
    response_bytes = recvall(sock, msglen)
    response = json.loads(response_bytes.decode('utf-8'))
    print("Response:", response)

    sock.close()

if __name__ == '__main__':
    main()
