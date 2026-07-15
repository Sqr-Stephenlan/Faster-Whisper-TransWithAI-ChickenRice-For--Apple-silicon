import plistlib
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "AI语音翻译.app"
ICON = APP / "Contents" / "Resources" / "AppIcon.icns"
INFO_PLIST = APP / "Contents" / "Info.plist"


def icns_chunk_types(path: Path) -> set[str]:
    data = path.read_bytes()
    if len(data) < 8 or data[:4] != b"icns":
        raise AssertionError(f"invalid ICNS header: {path}")

    declared_size = struct.unpack(">I", data[4:8])[0]
    if declared_size != len(data):
        raise AssertionError(f"invalid ICNS size: declared {declared_size}, actual {len(data)}")

    chunk_types: set[str] = set()
    offset = 8
    while offset + 8 <= len(data):
        chunk_type = data[offset : offset + 4].decode("ascii")
        chunk_size = struct.unpack(">I", data[offset + 4 : offset + 8])[0]
        if chunk_size < 8 or offset + chunk_size > len(data):
            raise AssertionError(f"invalid {chunk_type!r} chunk size: {chunk_size}")
        chunk_types.add(chunk_type)
        offset += chunk_size

    if offset != len(data):
        raise AssertionError(f"trailing ICNS data at offset {offset}")
    return chunk_types


class MacOSAppIconTests(unittest.TestCase):
    def test_icns_contains_standard_and_retina_representations(self) -> None:
        chunk_types = icns_chunk_types(ICON)

        self.assertTrue({"ic07", "ic08", "ic09"}.issubset(chunk_types))
        self.assertTrue({"ic10", "ic11", "ic12", "ic13", "ic14"}.issubset(chunk_types))

    def test_bundle_declares_icon_and_updated_build_number(self) -> None:
        with INFO_PLIST.open("rb") as file:
            info = plistlib.load(file)

        self.assertEqual(info["CFBundleIconFile"], "AppIcon.icns")
        self.assertEqual(info["CFBundleVersion"], "3")


if __name__ == "__main__":
    unittest.main()
