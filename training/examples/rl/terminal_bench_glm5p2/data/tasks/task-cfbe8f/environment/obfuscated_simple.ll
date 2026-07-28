; A single 8-bit addition obfuscated via MBA:
;   add i8 %a, %b  ->  (((a ^ b) + 2 * (a & b)) * 39 + 23) * 151 + 111

define i8 @simple_add(i8 %a, i8 %b) {
  %1 = xor i8 %b, %a
  %2 = and i8 %b, %a
  %3 = mul i8 2, %2
  %4 = add i8 %1, %3
  %5 = mul i8 39, %4
  %6 = add i8 23, %5
  %7 = mul i8 -105, %6
  %8 = add i8 111, %7
  ret i8 %8
}
