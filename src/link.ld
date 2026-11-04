OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 16M }
SECTIONS
{
    .text : {
        KEEP(*(.text.init))
        *(.text .text.*)
    } > RAM
    .rodata ALIGN(16) : {
        *(.rodata .rodata.* .srodata .srodata.*)
        *(.eh_frame .eh_frame.* .gcc_except_table .gcc_except_table.*)
    } > RAM
    .tohost ALIGN(64) : { KEEP(*(.tohost)) } > RAM
    .data ALIGN(16) : {
        *(.data .data.*)
        __global_pointer$ = . + 0x800;
        *(.sdata .sdata.*)
    } > RAM
    .preinit_array ALIGN(8) : {
        PROVIDE_HIDDEN(__preinit_array_start = .);
        KEEP(*(.preinit_array))
        PROVIDE_HIDDEN(__preinit_array_end = .);
    } > RAM
    .init_array ALIGN(8) : {
        PROVIDE_HIDDEN(__init_array_start = .);
        KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*)))
        KEEP(*(.init_array))
        PROVIDE_HIDDEN(__init_array_end = .);
    } > RAM
    .fini_array ALIGN(8) : {
        PROVIDE_HIDDEN(__fini_array_start = .);
        KEEP(*(SORT_BY_INIT_PRIORITY(.fini_array.*)))
        KEEP(*(.fini_array))
        PROVIDE_HIDDEN(__fini_array_end = .);
    } > RAM
    .bss ALIGN(16) (NOLOAD) : {
        _bss_start = .;
        *(.sbss .sbss.* .bss .bss.*)
        *(COMMON)
        . = ALIGN(16);
        _bss_end = .;
    } > RAM
    .heap ALIGN(16) (NOLOAD) : {
        __heap_start = .;
        PROVIDE(end = .);
        PROVIDE(_end = .);
        . = . + 0x10000;
        __heap_end = .;
    } > RAM
    .stack ALIGN(16) (NOLOAD) : {
        _stack_bottom = .;
        . = . + 0x10000;
        _stack_top = .;
    } > RAM
    ASSERT(_stack_top <= ORIGIN(RAM) + LENGTH(RAM), "RAM overflow")
}
