#!/usr/bin/env python3
"""Generate minimal BPF ELF relocatable objects from hex bytecode definitions.

Each output file is a valid ELF64 relocatable (ET_REL) with e_machine=EM_BPF(247),
containing a .text section with the BPF bytecode, plus .strtab and .symtab sections.
"""
import struct
import os
import sys


def hex_to_bytes(hex_str):
    return bytes.fromhex(hex_str)


def make_bpf_elf(bytecode, filename):
    """Create a minimal 64-bit little-endian ELF relocatable with BPF bytecode."""
    # String table (shared for section names and symbol names)
    strtab = b'\x00.text\x00.symtab\x00.strtab\x00prog\x00'
    NAME_TEXT = 1
    NAME_SYMTAB = 7
    NAME_STRTAB = 15
    NAME_PROG = 23

    ehdr_size = 64
    text_offset = ehdr_size
    text_size = len(bytecode)

    strtab_offset = text_offset + text_size
    strtab_size = len(strtab)

    # Align symtab to 8 bytes
    raw_symtab_off = strtab_offset + strtab_size
    symtab_padding = (8 - raw_symtab_off % 8) % 8
    symtab_offset = raw_symtab_off + symtab_padding

    # Symbol table: null + prog
    sym_null = struct.pack('<IBBHQQ', 0, 0, 0, 0, 0, 0)
    sym_prog = struct.pack('<IBBHQQ',
                           NAME_PROG,
                           (1 << 4) | 2,  # STB_GLOBAL | STT_FUNC
                           0, 1,          # st_other=0, st_shndx=1 (.text)
                           0, text_size)  # st_value=0, st_size
    symtab = sym_null + sym_prog
    symtab_size = len(symtab)

    # Section headers follow symtab, aligned to 8
    raw_shdr_off = symtab_offset + symtab_size
    shdr_padding = (8 - raw_shdr_off % 8) % 8
    shdr_offset = raw_shdr_off + shdr_padding

    # --- ELF header ---
    e_ident = b'\x7fELF'
    e_ident += bytes([2, 1, 1, 0])  # class=64, data=LE, version=1, osabi=0
    e_ident += b'\x00' * 8          # padding
    assert len(e_ident) == 16

    ehdr = e_ident
    ehdr += struct.pack('<HHIQQQIHHHHHH',
                        1,              # e_type: ET_REL
                        247,            # e_machine: EM_BPF
                        1,              # e_version
                        0,              # e_entry
                        0,              # e_phoff
                        shdr_offset,    # e_shoff
                        0,              # e_flags
                        64,             # e_ehsize
                        0,              # e_phentsize
                        0,              # e_phnum
                        64,             # e_shentsize
                        4,              # e_shnum (null + .text + .strtab + .symtab)
                        2)              # e_shstrndx (.strtab = section 2)
    assert len(ehdr) == 64

    # --- Section headers (64 bytes each) ---
    def shdr(name, typ, flags, offset, size, link=0, info=0, align=0, entsize=0):
        return struct.pack('<IIQQQQIIQQ',
                           name, typ, flags, 0, offset, size,
                           link, info, align, entsize)

    shdrs = b''
    shdrs += shdr(0, 0, 0, 0, 0)                                         # NULL
    shdrs += shdr(NAME_TEXT, 1, 6, text_offset, text_size, align=8)       # .text PROGBITS
    shdrs += shdr(NAME_STRTAB, 3, 0, strtab_offset, strtab_size, align=1)  # .strtab
    shdrs += shdr(NAME_SYMTAB, 2, 0, symtab_offset, symtab_size,
                  link=2, info=1, align=8, entsize=24)                    # .symtab

    with open(filename, 'wb') as f:
        f.write(ehdr)
        f.write(bytecode)
        f.write(strtab)
        if symtab_padding:
            f.write(b'\x00' * symtab_padding)
        f.write(symtab)
        if shdr_padding:
            f.write(b'\x00' * shdr_padding)
        f.write(shdrs)


# ---------------------------------------------------------------------------
# BPF program definitions: (hex instruction strings)
# Instruction encoding: 8 bytes LE per insn, 16 hex chars
#   byte 0: opcode
#   byte 1: (src_reg<<4)|dst_reg
#   bytes 2-3: offset (s16 LE)
#   bytes 4-7: imm (s32 LE)
# ---------------------------------------------------------------------------

PROGRAMS = {
    # --- Basic programs (1-15) ---

    # 1: ACCEPT - mov r0, 0; exit
    "prog_01.o": ["b700000000000000", "9500000000000000"],

    # 2: REJECT - mov r0, r2 (r2 NOT_INIT); exit
    "prog_02.o": ["bf20000000000000", "9500000000000000"],

    # 3: REJECT - mov r1, 5 (r0 never set); exit
    "prog_03.o": ["b701000005000000", "9500000000000000"],

    # 4: REJECT - mov r0, 0x42; stx_dw [r10-520], r0 (oob); exit
    "prog_04.o": ["b700000042000000", "7b0af8fd00000000", "9500000000000000"],

    # 5: ACCEPT - mov r0, 7; stx_dw [r10-8], r0; ldx_dw r0, [r10-8]; exit
    "prog_05.o": ["b700000007000000", "7b0af8ff00000000",
                   "79a0f8ff00000000", "9500000000000000"],

    # 6: REJECT - ldx_dw r0, [r10-8] (uninit stack); exit
    "prog_06.o": ["79a0f8ff00000000", "9500000000000000"],

    # 7: REJECT - mov r0, 0; ja +1; mov r0, 1 (unreachable); exit
    "prog_07.o": ["b700000000000000", "0500010000000000",
                   "b700000001000000", "9500000000000000"],

    # 8: REJECT - mov r0, 0; mov r2, 0x1000; stx_w [r2+0], r0 (scalar base); exit
    "prog_08.o": ["b700000000000000", "b702000000100000",
                   "6302000000000000", "9500000000000000"],

    # 9: ACCEPT - mov r0, 0; jeq r0, 0, +1; mov r0, 1; exit
    "prog_09.o": ["b700000000000000", "1500010000000000",
                   "b700000001000000", "9500000000000000"],

    # 10: REJECT - mov r0, 0; ja -2 (back-edge to insn 0); exit
    "prog_10.o": ["b700000000000000", "0500feff00000000", "9500000000000000"],

    # 11: REJECT - mov r0, 0; ja +100 (oob target); exit
    "prog_11.o": ["b700000000000000", "0500640000000000", "9500000000000000"],

    # 12: ACCEPT - mov r6, 1; call #1; mov r0, r6 (callee-saved); exit
    "prog_12.o": ["b706000001000000", "8500000001000000",
                   "bf60000000000000", "9500000000000000"],

    # 13: REJECT - mov r0, 0; call #1; mov r0, r1 (r1 NOT_INIT after call); exit
    "prog_13.o": ["b700000000000000", "8500000001000000",
                   "bf10000000000000", "9500000000000000"],

    # 14: REJECT - mov r10, 0 (R10 immutable); mov r0, 0; exit
    "prog_14.o": ["b70a000000000000", "b700000000000000", "9500000000000000"],

    # 15: ACCEPT - ldx_w r0, [r1+0] (ctx access); exit
    "prog_15.o": ["6110000000000000", "9500000000000000"],

    # --- Bug-triggering programs (16-20) ---

    # 16: ACCEPT - call #1; add64 r0, 1 (uses R0 return value); exit
    #   Correct: R0=SCALAR after call, add is fine -> ACCEPT
    "prog_16.o": ["8500000001000000", "0700000001000000", "9500000000000000"],

    # 17: REJECT - mov r0, 0; add64 r10, 8 (ALU write to R10); exit
    #   Correct: R10 is immutable, any write forbidden -> REJECT
    "prog_17.o": ["b700000000000000", "070a000008000000", "9500000000000000"],

    # 18: REJECT - mov r0, 0; jne32 r0, 0, +1; mov r2, 1; mov r0, r2; exit
    #   Correct: taken path (0->1->3) reads uninit R2 -> REJECT
    "prog_18.o": ["b700000000000000", "5600010000000000",
                   "b702000001000000", "bf20000000000000", "9500000000000000"],

    # 19: REJECT - mov r0, 7; stx_dw [r10-1], r0 (offset+size overflows); exit
    #   Correct: offset=-1, size=8, -1+8=7>0 -> stack overflow -> REJECT
    "prog_19.o": ["b700000007000000", "7b0affff00000000", "9500000000000000"],

    # 20: ACCEPT - mov r0, 7; stx_dw [r10-512], r0; exit
    #   Correct: offset=-512, size=8, -512+8=-504<=0 -> valid boundary -> ACCEPT
    "prog_20.o": ["b700000007000000", "7b0a00fe00000000", "9500000000000000"],
}


if __name__ == '__main__':
    outdir = sys.argv[1] if len(sys.argv) > 1 else '/app/programs'
    os.makedirs(outdir, exist_ok=True)
    for name, hex_insns in sorted(PROGRAMS.items()):
        bytecode = b''.join(hex_to_bytes(h) for h in hex_insns)
        make_bpf_elf(bytecode, os.path.join(outdir, name))
    print(f"Generated {len(PROGRAMS)} BPF ELF objects in {outdir}")
