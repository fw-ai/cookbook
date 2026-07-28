; Runtime equivalence test: add_four(3, 5, 7, 2) should return 17.

target triple = "x86_64-unknown-linux-gnu"

define i8 @add_four(i8 %a, i8 %b, i8 %c, i8 %d) {
  ; MBA(a, b) = a + b
  %1 = xor i8 %b, %a
  %2 = and i8 %b, %a
  %3 = mul i8 2, %2
  %4 = add i8 %1, %3
  %5 = mul i8 39, %4
  %6 = add i8 23, %5
  %7 = mul i8 -105, %6
  %8 = add i8 111, %7
  ; MBA((a+b), c)
  %9 = xor i8 %8, %c
  %10 = and i8 %8, %c
  %11 = mul i8 2, %10
  %12 = add i8 %9, %11
  %13 = mul i8 39, %12
  %14 = add i8 23, %13
  %15 = mul i8 -105, %14
  %16 = add i8 111, %15
  ; MBA(((a+b)+c), d)
  %17 = xor i8 %16, %d
  %18 = and i8 %16, %d
  %19 = mul i8 2, %18
  %20 = add i8 %17, %19
  %21 = mul i8 39, %20
  %22 = add i8 23, %21
  %23 = mul i8 -105, %22
  %24 = add i8 111, %23
  ret i8 %24
}

define i32 @main() {
  %1 = call i8 @add_four(i8 3, i8 5, i8 7, i8 2)
  %2 = sext i8 %1 to i32
  ret i32 %2
}
