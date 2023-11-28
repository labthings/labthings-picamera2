from __future__ import annotations
import logging
from labthings_fastapi.thing_server import ThingServer
from .thing import StreamingPiCamera2


logging.basicConfig(level=logging.INFO)

thing_server = ThingServer()
my_thing = StreamingPiCamera2()
my_thing.validate_thing_description()
thing_server.add_thing(my_thing, "/camera")

app = thing_server.app
