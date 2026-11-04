.section .text.init,"ax",@progbits
.globl _start
.type _start,@function
_start:
    .option push
    .option norelax
    la gp, __global_pointer$
    la sp, _stack_top
    .option pop
    la t0, _bss_start
    la t1, _bss_end
1:
    bgeu t0, t1, 2f
    sd zero, 0(t0)
    addi t0, t0, 8
    j 1b
2:
    call __libc_init_array
    li a0, 0
    li a1, 0
    call main
    call _exit
3:
    j 3b
.size _start, .-_start

.section .tohost,"aw",@progbits
.balign 64
.globl tohost
.type tohost,@object
tohost:
    .dword 0
.size tohost, 8
.balign 64
.globl fromhost
.type fromhost,@object
fromhost:
    .dword 0
.size fromhost, 8

.section .note.GNU-stack,"",@progbits
