.section .text

.global usefulGadgets
.global trickGadgets

/*
 * usefulGadgets - ROP gadgets for chain construction
 * Offsets (from usefulGadgets symbol):
 *   +0:  pop rax; ret
 *   +2:  xchg rax, rsp; ret
 *   +5:  mov rax, [rax]; ret
 *   +9:  add rax, rbp; ret
 *   +13: pop rdi; ret
 *   +15: pop rsi; pop r15; ret
 *   +19: pop rbp; ret
 */
usefulGadgets:
    popq %rax
    ret
    xchg %rax, %rsp
    ret
    movq (%rax), %rax
    ret
    addq %rbp, %rax
    ret
    popq %rdi
    ret
    popq %rsi
    popq %r15
    ret
    popq %rbp
    ret

/*
 * trickGadgets - indirect register manipulation
 * Offsets (from trickGadgets symbol):
 *   +0:  mov rdx, rbp; pop rbp; ret
 *   +5:  jmp *rax
 *   +7:  nop; ret
 */
trickGadgets:
    movq %rbp, %rdx
    popq %rbp
    ret
    jmpq *%rax
    nop
    ret

/* Mark this object as not requiring executable stack */
.section .note.GNU-stack,"",@progbits
