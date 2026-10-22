"""Reject ambiguous/mismatched ARM and MIPS ABIs instead of guessing pointer sizes."""
import struct
import hashlib

import pytest

from test_ida2llvm_microcode import L


def header(bits=32, endian="little", machine=40, flags=0x05000200, kind=3):
    ident = b"\x7fELF" + bytes([1 if bits == 32 else 2, 1 if endian == "little" else 2, 1]) + bytes(9)
    fmt = ("<" if endian == "little" else ">") + ("HHIIIIIHHHHHH" if bits == 32 else "HHIQQQIHHHHHH")
    return ident + struct.pack(fmt, kind, machine, 1, 0, 0, 0, flags, 52 if bits == 32 else 64, 0, 0, 0, 0, 0)


@pytest.mark.parametrize("bits,endian", [(32, "little"), (32, "big"), (64, "little"), (64, "big")])
def test_elf_fixed_header(L, bits, endian):
    result = L._parse_elf_header(header(bits, endian, 8, 0x70000000))
    assert result == dict(bits=bits, endian=endian, machine=8, flags=0x70000000, type=3)


@pytest.mark.parametrize("raw", [b"", b"not ELF", header()[:30], b"\x7fELF" + bytes(60)])
def test_bad_header_rejected(L, raw):
    with pytest.raises(L.UnsupportedTarget):
        L._parse_elf_header(raw)


@pytest.mark.parametrize("flags,abi", [(0x05000200, "aapcs-base"), (0x05000400, "aapcs-vfp")])
def test_arm_linked_abi(L, flags, abi):
    spec = L._apply_source_abi(L.source_target_spec("ARM", 32, False), L._parse_elf_header(header(flags=flags)))
    assert spec["abi"] == abi
    assert spec["triple"].endswith("gnueabihf" if abi == "aapcs-vfp" else "gnueabi")


@pytest.mark.parametrize("flags,kind", [(0x05000600, 3), (0x04000400, 3), (0x05000000, 1), (0x05000000, 2), (0x05000000, 3)])
def test_arm_ambiguous_abi_rejected(L, flags, kind):
    with pytest.raises(L.UnsupportedTarget):
        L._apply_source_abi(L.source_target_spec("ARM", 32, False), L._parse_elf_header(header(flags=flags, kind=kind)))


@pytest.mark.parametrize("bits,flags", [(32, 0x20), (64, 0x20), (32, 0x2000), (64, 0x1000), (32, 0x3000)])
def test_mips_unsupported_abi_rejected(L, bits, flags):
    spec = L.source_target_spec("mipsb", bits, True)
    with pytest.raises(L.UnsupportedTarget):
        L._apply_source_abi(spec, L._parse_elf_header(header(bits, "big", 8, flags)))


@pytest.mark.parametrize("bits,flags,abi", [(32, 0x1000, "o32"), (32, 0, "o32"), (64, 0, "n64")])
def test_mips_supported_abi(L, bits, flags, abi):
    spec = L.source_target_spec("mipsb", bits, True)
    result = L._apply_source_abi(spec, L._parse_elf_header(header(bits, "big", 8, flags)))
    assert result["abi"] == abi


def test_elf_ida_mismatch_rejected(L):
    spec = L.source_target_spec("mipsb", 64, True)
    with pytest.raises(L.UnsupportedTarget, match="class/endianness"):
        L._apply_source_abi(spec, L._parse_elf_header(header(32, "big", 8, 0x1000)))


def test_mapped_idb_header_needs_no_original_file(L, monkeypatch):
    monkeypatch.setattr(L.ida_nalt, "get_imagebase", lambda: 0x400000)
    monkeypatch.setattr(L.ida_bytes, "get_bytes", lambda ea, size: header())
    result = L._source_elf_header()
    assert result["evidence"] == "idb-image-header"


@pytest.mark.parametrize("matching", [True, False])
def test_source_file_fingerprint(L, monkeypatch, tmp_path, matching):
    binary = tmp_path / "source.elf"
    raw = header()
    binary.write_bytes(raw)
    monkeypatch.setattr(L.ida_nalt, "get_imagebase", lambda: 0)
    monkeypatch.setattr(L.ida_bytes, "get_bytes", lambda ea, size: None)
    monkeypatch.setattr(L.ida_nalt, "get_input_file_path", lambda: str(binary))
    monkeypatch.setattr(L.ida_nalt, "retrieve_input_file_sha256", lambda: hashlib.sha256(raw if matching else b"other").digest())
    if matching:
        assert L._source_elf_header()["evidence"] == "verified-source-file"
    else:
        with pytest.raises(L.UnsupportedTarget, match="fingerprint differs"):
            L._source_elf_header()
