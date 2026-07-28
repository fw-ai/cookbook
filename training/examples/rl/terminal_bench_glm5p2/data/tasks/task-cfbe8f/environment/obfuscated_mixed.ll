; Mix of MBA-obfuscated i8 add and normal i32 arithmetic.
; The pass should deobfuscate only the MBA pattern and leave everything else alone.

define i32 @mixed_ops(i8 %a, i8 %b, i32 %x, i32 %y) {
  ; MBA(a, b) — should be deobfuscated
  %1 = xor i8 %b, %a
  %2 = and i8 %b, %a
  %3 = mul i8 2, %2
  %4 = add i8 %1, %3
  %5 = mul i8 39, %4
  %6 = add i8 23, %5
  %7 = mul i8 -105, %6
  %8 = add i8 111, %7
  ; Normal i32 add — must NOT be touched
  %9 = add i32 %x, %y
  ; Combine both results
  %10 = sext i8 %8 to i32
  %11 = add i32 %10, %9
  ret i32 %11
}
