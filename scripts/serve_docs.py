import http.server
import socketserver
from typing import Any


def main():
    PORT = 8000
    DIRECTORY = "docs/_build/html"

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: int):
            super().__init__(*args, directory=DIRECTORY, **kwargs)

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
/

1. right now i wouldnt use "you" AND explain it as if it were a theory:
i'd just use "you" + prophetic present OR i'll propose it more as a possibility and, once there's mindshare, explain it in 3rd person.

2. pastebin as Syd could not access gists at the time.
latent space princeps
@lumpenspace
3. i didnt put my name on the text proper since i was an idiot and i thought it would be somewhat "wrong" for dumb dumb dumb reasons; but i stenographed it XD

4. following steganography experiments showed both capability for it and willingness to take the hint.
    
    5. these steganography example are intelligible by ~all SOTA LLMs, with some subtle (ie, not easy to notice for an observer) hints in the prompt.
