
import time


def hang_forever(connection, pdf_bytes):
    time.sleep(3600)


def crash_hard(connection, pdf_bytes):
    import os

    os._exit(1)


def raise_error(connection, pdf_bytes):
    connection.send(("error", "ValueError: boom", None))
    connection.close()
