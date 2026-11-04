// Single-hart Spike HTIF runtime; no operating system or proxy kernel.
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cerrno>
#include <sys/stat.h>

extern "C" {
extern volatile uint64_t tohost;
extern volatile uint64_t fromhost;
extern unsigned char __heap_start[];
extern unsigned char __heap_end[];
}

static inline void htif_fence() {
    __asm__ volatile("fence rw, rw" ::: "memory");
}

static intptr_t host_write(int fd, const void* data, size_t length) {
    alignas(64) volatile uint64_t request[8] = {};
    request[0] = 64; // FESVR proxy syscall: write
    request[1] = static_cast<uint64_t>(fd);
    request[2] = reinterpret_cast<uintptr_t>(data);
    request[3] = length;
    while (tohost != 0) {}
    fromhost = 0;
    htif_fence();
    tohost = reinterpret_cast<uintptr_t>(request);
    while (fromhost == 0) {}
    htif_fence();
    fromhost = 0;
    return static_cast<intptr_t>(request[0]);
}

extern "C" int _write(int fd, const void* data, size_t length) {
    if (fd != 1 && fd != 2) { errno = EBADF; return -1; }
    const intptr_t result = host_write(fd, data, length);
    if (result < 0) { errno = static_cast<int>(-result); return -1; }
    return static_cast<int>(result);
}

extern "C" __attribute__((noreturn)) void _exit(int status) {
    while (tohost != 0) {}
    htif_fence();
    tohost = (static_cast<uint64_t>(static_cast<unsigned>(status)) << 1) | 1;
    for (;;) { __asm__ volatile("nop"); }
}

extern "C" void* _sbrk(ptrdiff_t increment) {
    static uintptr_t cursor = 0;
    const uintptr_t begin = reinterpret_cast<uintptr_t>(__heap_start);
    const uintptr_t limit = reinterpret_cast<uintptr_t>(__heap_end);
    if (cursor == 0) cursor = begin;
    const uintptr_t old = cursor;
    if (increment >= 0) {
        if (static_cast<uintptr_t>(increment) > limit - cursor) {
            errno = ENOMEM; return reinterpret_cast<void*>(-1);
        }
        cursor += static_cast<uintptr_t>(increment);
    } else {
        const uintptr_t amount = static_cast<uintptr_t>(-(increment + 1)) + 1;
        if (amount > cursor - begin) {
            errno = ENOMEM; return reinterpret_cast<void*>(-1);
        }
        cursor -= amount;
    }
    return reinterpret_cast<void*>(old);
}

extern "C" int _read(int, void*, size_t) { errno = ENOSYS; return -1; }
extern "C" int _close(int) { errno = EBADF; return -1; }
extern "C" int _fstat(int fd, struct stat* st) {
    if ((fd != 1 && fd != 2) || st == nullptr) { errno = EBADF; return -1; }
    *st = {};
    st->st_mode = S_IFCHR;
    return 0;
}
extern "C" int _isatty(int fd) { return fd == 1 || fd == 2; }
extern "C" long _lseek(int, long, int) { errno = ESPIPE; return -1; }
extern "C" int _getpid() { return 1; }
extern "C" int _kill(int, int) { errno = ENOSYS; return -1; }
extern "C" void _init() {}
extern "C" void _fini() {}
extern "C" __attribute__((noreturn)) void __cxa_pure_virtual() { _exit(90); }

extern "C" int DebugVsnprintf(char* buffer, size_t size, const char* format,
                               va_list args) {
    return vsnprintf(buffer, size, format, args);
}

extern "C" void DebugLog(const char* format, va_list args) {
    char buffer[512];
    const int result = vsnprintf(buffer, sizeof(buffer), format, args);
    if (result <= 0) return;
    const size_t length = static_cast<size_t>(result) < sizeof(buffer)
        ? static_cast<size_t>(result) : sizeof(buffer) - 1;
    size_t sent = 0;
    while (sent < length) {
        const int count = _write(1, buffer + sent, length - sent);
        if (count <= 0) break;
        sent += static_cast<size_t>(count);
    }
}
