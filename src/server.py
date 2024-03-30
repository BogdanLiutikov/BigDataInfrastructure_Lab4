import json
from configparser import ConfigParser
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from socket import socket
from socketserver import BaseServer
from typing import Any
from predict import Predictor


predictor = Predictor()


class RequestHandler(BaseHTTPRequestHandler):
    def __init__(self, request: socket | tuple[bytes, socket], client_address: Any, server: BaseServer) -> None:
        super().__init__(request, client_address, server)

    def do_GET(self):
        print('Пришел get запрос')
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        result = json.dumps({'Message': 'OK'}).encode('utf-8')
        self.wfile.write(result)

    def do_POST(self):
        print('Пришел post запрос')
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        json_data = json.loads(body)
        predictions = predictor.predict(json_data['x'])

        response = BytesIO()
        result = json.dumps({'y': predictions}).encode('utf-8')
        response.write(result)
        self.wfile.write(response.getvalue())


class Server:
    def __init__(self, adress: str, port: int) -> None:
        self.adress = adress
        self.port = port
        self.handler = RequestHandler

    def run(self):
        httpd = HTTPServer((self.adress, self.port), self.handler)
        print("Запуск сервера")
        httpd.serve_forever()


if __name__ == "__main__":
    config = ConfigParser()
    config.read('config.ini')
    adress = config.get('server', 'adress')
    port = config.getint('server', 'port')
    server = Server(adress, port)
    server.run()
