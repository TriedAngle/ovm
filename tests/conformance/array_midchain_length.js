// ES 10.4.2.2: an Array's `length` is an own data property, so
// OrdinaryGet (ES 10.1.8.1) must find it when walking *through* an
// array, not only when the array is the receiver. Object.create is used
// because the literal `__proto__:` form is currently a no-op (#7).
var proto = [1, 2, 3];
var o = Object.create(proto);
if (o.length !== 3) {
    throw new TypeError("mid-chain array length: expected 3, got " + o.length);
}
