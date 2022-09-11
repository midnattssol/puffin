#!/usr/bin/env python3.10
"""Find out if a file has changed."""
import dataclasses as dc
import hashlib as hl
import pathlib as p

# ===| Globals |===
CHUNK = 512

# ===| Classes |===


@dc.dataclass
class FileStatus:
    """Used to check if a file has changed."""

    filename: p.Path
    length: int = None
    sha256: int = None
    first_chunk: bytes = None

    def __post_init__(self):
        self.filename = p.Path(self.filename)
        self.update()

    def update(self):
        """Update the variables by reading the file in chunks once."""
        self.length = 0
        sha256hasher = hl.sha256()
        self.first_chunk = b""

        first_chunk_read = False

        with self.filename.open("rb") as file:
            while text := file.read(CHUNK):
                sha256hasher.update(text)
                self.length += len(text)
                self.first_chunk = text if not first_chunk_read else self.first_chunk

                first_chunk_read = True

        self.sha256 = sha256hasher.digest()

    def changed(self, update: bool = True) -> bool:
        """Lazily check if a file has changed.

        The function is chunked, so it's safe for use with arbitrarily large files.
        It also updates the filestatus unless the `update` flag is set to False.
        """
        file_length = 0
        has_read_first = False
        hasher = hl.sha256()

        # Load the file chunk by chunk.
        with self.filename.open("rb") as file:
            while text := file.read(CHUNK):

                # Check if the first chunk is the same.
                if not has_read_first and text != self.first_chunk:
                    if update:
                        self.update()
                    return True

                has_read_first = True
                file_length += len(text)
                hasher.update(text)

                # Check if the file is too long.
                if file_length > self.length:
                    if update:
                        self.update()
                    return True

        if file_length != self.length:
            if update:
                self.update()
            return True

        sha256 = hasher.digest()
        if self.sha256 != sha256:
            if update:
                self.update()
            return True

        return False


if __name__ == "__main__":
    this_file = FileStatus(p.Path(__file__))
    this_file.first_chunk = ...

    assert this_file.changed()
    assert not this_file.changed()
