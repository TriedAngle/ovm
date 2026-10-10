# Conventions

## Method vs namespace function

Can the function trigger a GC (does it need `&mut Heap`, or take `Handle`s because it allocates)?

- **No GC** (only `&Heap` or nothing besides the receiver): make it a method.
  `obj.map(heap)`, `v.is_truthy(heap)`, `key.classify_key(heap)`
- **Can GC**: keep it a namespace function. `heap` always comes first, then the
  objects it operates on.
  `Transition::define_own_property(heap, scope, receiver, name, desc)`,
  `Convert::to_string(heap, scope, v)`

Cross-cutting notifications keep their domain namespace even when GC-free
(e.g. `Prototype::shape_changed(heap, map)` — it notifies the prototype
registry, it is not an operation on the map).

Only `Tagged` auto-derefs to its pointee. `Handle` has no `Deref` on purpose: a
`&self` behind a handle deref could dangle once the GC moves the object, so
GC-able operations take handles explicitly.
