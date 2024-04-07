from http.server import BaseHTTPRequestHandler, HTTPServer


class MockServerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()

        self.wfile.write(b"TERMINATE_ON_HOST_MAINTENANCE")


def run(server_class=HTTPServer, handler_class=MockServerHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting mock server on port {port}...')
    httpd.serve_forever()


if __name__ == '__main__':
    run()
