use core::cell::UnsafeCell;
use core::ptr::NonNull;

use crate::{
    CallableInfoInit, CallableInfoObject, Context, ContextInit, FixedArray, FixedByteArray, Global,
    Handle, HandleData, HandleScope, Heap, Map, MapInit, MapKind, Object, ObjectInit, RootHandles,
    ScopeInfo, ScopeInfoInit, SlotName, Smi, StringInterner, Symbol, Tagged, Value,
};

#[derive(Clone, Copy)]
pub struct WellKnown {
    // vm internal machinery: sentinels, canonical empties, builtin maps
    pub map_map: Global<Map>,
    pub the_hole: Global<Object>,
    /// Never user-visible: a call returning it signals a pending exception.
    pub exception: Global<Object>,
    /// exception map
    pub exception_map: Global<Map>,
    /// Shared empty backing for objects without data slots (never written in
    /// place; the first property store swaps in a fresh array).
    pub empty_fixed_array: Global<FixedArray>,
    /// TODO: Placeholder for now
    pub empty_context: Global<Context>,
    pub smi_map: Global<Map>,
    pub float_map: Global<Map>,
    pub array_map: Global<Map>,
    pub byte_array_map: Global<Map>,
    pub string_map: Global<Map>,
    pub symbol_map: Global<Map>,
    pub accessor_pair_map: Global<Map>,
    pub callable_map: Global<Map>,
    pub handler_table_map: Global<Map>,
    pub context_map: Global<Map>,
    pub scope_info_map: Global<Map>,
    /// Shared immutable scope description for contexts without named slots
    /// (block/catch contexts, the empty context)
    pub empty_scope_info: Global<ScopeInfo>,
    // js userspace primitives and object base maps
    pub undefined: Global<Object>,
    pub undefined_map: Global<Map>,
    pub null: Global<Object>,
    pub null_map: Global<Map>,
    pub false_object: Global<Object>,
    pub true_object: Global<Object>,
    pub boolean_map: Global<Map>,
    /// Base map for ECMAScript array objects (elements + length, prototype
    /// %Array.prototype%).
    pub js_array_map: Global<Map>,
    /// Base map for ECMAScript error objects
    pub error_map: Global<Map>,
    /// Per-class error maps (prototype chain carries `.constructor`);
    /// installed by the builtins bootstrap.
    pub type_error_map: Global<Map>,
    pub reference_error_map: Global<Map>,
    pub range_error_map: Global<Map>,
    /// Wrapper maps for boxed primitives (slots[0] = the primitive value);
    /// installed by the builtins bootstrap.
    pub number_wrapper_map: Global<Map>,
    pub boolean_wrapper_map: Global<Map>,
    pub string_wrapper_map: Global<Map>,
    // prototypes
    /// `%Object.prototype%`: root of the ordinary-object prototype hierarchy.
    pub object_prototype: Global<Object>,
    /// `%Array.prototype%`: an array object, parent of all array instance maps.
    pub array_prototype: Global<Object>,
    /// `%Error.prototype%`: parent of all error instance maps.
    pub error_prototype: Global<Object>,
    /// `%Function.prototype%` (ES 19.2.3): the canonical empty function — a
    /// callable, non-constructable function whose `[[Prototype]]` is
    /// `%Object.prototype%`; the `[[Prototype]]` of ordinary function objects.
    pub function_prototype: Global<Object>,
    /// The realm global object: global variables are properties on it
    /// (top-level `var`/assignments; lexical script-context globals later).
    pub global_object: Global<Object>,
    /// Initial map of `%Object.prototype%`: every fresh `{}` gets it
    /// (extendable, parent = object_prototype). Shared with `global_object`.
    pub object_initial_map: Global<Map>,
    /// Function object map
    /// slots[0] = shared CallableInfoObject, slots[1] = closure context.
    pub function_map: Global<Map>,
    /// Callable function objects without [[Construct]] (arrows, methods,
    /// getters and setters).
    pub non_constructor_function_map: Global<Map>,
    /// Constructible class functions whose ordinary [[Call]] path throws.
    pub class_constructor_map: Global<Map>,
    /// The `@@toPrimitive` well-known symbol
    pub to_primitive_symbol: Global<Symbol>,
    /// The `@@iterator` well-known symbol (internal: no Symbol global yet)
    pub iterator_symbol: Global<Symbol>,
    /// Map of array-iterator objects (slots: [iterated array, next index])
    pub array_iterator_map: Global<Map>,
    /// %ArrayIteratorPrototype% (holds `next` and @@iterator)
    pub array_iterator_prototype: Global<Object>,
    /// Map of iterator-result objects `{ value, done }` (w+, e+, c+)
    pub iterator_result_map: Global<Map>,
    /// Hidden for-in enumerator: slots [level, keys, index, visited]
    /// (ES 14.7.5.9 EnumerateObjectProperties; unreachable from JS)
    pub for_in_enumerator_map: Global<Map>,
    pub strings: WellKnownStrings,
}

macro_rules! define_well_known_strings {
    ($($field:ident => $text:literal),* $(,)?) => {
        #[derive(Debug, Clone, Copy)]
        pub struct WellKnownStrings {
            $(pub $field: Global<SlotName>,)*
        }

        impl WellKnownStrings {
            pub fn uninit(roots: &RootHandles) -> Self {
                Self {
                    $($field: unsafe { smi_handle::<SlotName>(roots) },)*
                }
            }

            pub fn intern_all(
                heap: &mut Heap,
                interner: &StringInterner,
                roots: &RootHandles,
            ) -> Self {
                Self {
                    $($field: {
                        let interned = interner.intern(heap, roots, $text);
                        roots.create_handle(SlotName::from(interned.as_tagged()).tagged())
                    },)*
                }
            }
        }
    };
}

define_well_known_strings! {
    empty => "",
    length => "length",
    name => "name",
    message => "message",
    prototype => "prototype",
    constructor => "constructor",
    to_string => "toString",
    value_of => "valueOf",
    next => "next",
    done => "done",
    value => "value",
    values => "values",
    default => "default",
    number => "number",
    string => "string",
    boolean => "boolean",
    number_ctor => "Number",
    boolean_ctor => "Boolean",
    symbol_ctor => "Symbol",
    object => "object",
    function => "function",
    symbol => "symbol",
    undefined => "undefined",
    null => "null",
    true_ => "true",
    false_ => "false",
}

unsafe fn smi_handle<T>(roots: &RootHandles) -> Global<T> {
    roots.create_handle(unsafe { Tagged::from_value_unchecked(Smi::new(0).encode()) })
}

impl WellKnown {
    /// Placeholder set: every slot is a smi sentinel handle. Valid to
    /// install before the real bootstrap phases run.
    pub fn uninit(roots: &RootHandles) -> Self {
        uninited_wellknown(roots)
    }
}

fn uninited_wellknown(roots: &RootHandles) -> WellKnown {
    let map = unsafe { smi_handle::<Map>(roots) };
    let obj = unsafe { smi_handle::<Object>(roots) };
    let array = unsafe { smi_handle::<FixedArray>(roots) };
    let context = unsafe { smi_handle::<Context>(roots) };
    let scope_info = unsafe { smi_handle::<ScopeInfo>(roots) };
    WellKnown {
        map_map: map,
        the_hole: obj,
        exception: obj,
        exception_map: map,
        empty_fixed_array: array,
        empty_context: context,
        smi_map: map,
        float_map: map,
        array_map: map,
        byte_array_map: map,
        string_map: map,
        symbol_map: map,
        accessor_pair_map: map,
        callable_map: map,
        handler_table_map: map,
        context_map: map,
        scope_info_map: map,
        empty_scope_info: scope_info,
        undefined: obj,
        undefined_map: map,
        null: obj,
        null_map: map,
        false_object: obj,
        true_object: obj,
        boolean_map: map,
        js_array_map: map,
        error_map: map,
        type_error_map: map,
        reference_error_map: map,
        range_error_map: map,
        number_wrapper_map: map,
        boolean_wrapper_map: map,
        string_wrapper_map: map,
        object_prototype: obj,
        array_prototype: obj,
        error_prototype: obj,
        function_prototype: obj,
        global_object: obj,
        object_initial_map: map,
        function_map: map,
        non_constructor_function_map: map,
        class_constructor_map: map,
        to_primitive_symbol: unsafe { smi_handle::<Symbol>(roots) },
        iterator_symbol: unsafe { smi_handle::<Symbol>(roots) },
        array_iterator_map: map,
        array_iterator_prototype: obj,
        iterator_result_map: map,
        for_in_enumerator_map: map,
        strings: WellKnownStrings::uninit(roots),
    }
}

fn alloc_map(heap: &mut Heap, roots: &RootHandles, kind: MapKind) -> Global<Map> {
    heap.allocate::<Map>(MapInit {
        kind,
        value_slot_count: 0,
        descriptors: &[],
        prototype: heap.known().null.erase(),
    })
    .into_global(roots)
}

fn alloc_parent_map(
    heap: &mut Heap,
    roots: &RootHandles,
    kind: MapKind,
    parent: Global<Object>,
) -> Global<Map> {
    alloc_parent_map_with_slots(heap, roots, kind, parent, 0)
}

fn alloc_parent_map_with_slots(
    heap: &mut Heap,
    roots: &RootHandles,
    kind: MapKind,
    parent: Global<Object>,
    value_slot_count: usize,
) -> Global<Map> {
    heap.allocate::<Map>(MapInit {
        kind,
        value_slot_count,
        descriptors: &[],
        prototype: parent.erase(),
    })
    .into_global(roots)
}

fn alloc_object(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    roots: &RootHandles,
    map: Global<Map>,
) -> Global<Object> {
    heap.new_object(scope, map, &[]).into_global(roots)
}

pub fn bootstrap_basics(heap: &mut Heap, roots: &RootHandles) {
    let mut known = uninited_wellknown(roots);
    heap.set_known(known);

    let map_map = heap
        .allocate::<Map>(MapInit {
            kind: MapKind::MAP,
            value_slot_count: 0,
            descriptors: &[],
            prototype: roots.create_handle(Smi::new(0).encode()),
        })
        .into_global(roots);
    heap.no_gc(|nogc| {
        map_map
            .heap_ref(nogc)
            .header
            .map
            .set(nogc, map_map, map_map);
    });
    known.map_map = map_map;
    heap.set_known(known);

    let data = HandleData::new(Smi::new(0).encode());
    let scope = unsafe { HandleScope::from_raw(NonNull::from(&data)) };

    let the_hole_map = heap
        .allocate::<Map>(MapInit {
            kind: MapKind::OBJECT,
            value_slot_count: 0,
            descriptors: &[],
            prototype: scope.handle(Smi::new(0)),
        })
        .into_global(roots);
    let the_hole = heap
        .allocate::<Object>(ObjectInit {
            map: the_hole_map,
            slots: unsafe { smi_handle::<FixedArray>(roots) },
            elements: unsafe { smi_handle::<Value>(roots) },
            length: 0,
        })
        .into_global(roots);

    let null_map: Handle<'_, Map> = heap
        .allocate::<Map>(MapInit {
            kind: MapKind::OBJECT,
            value_slot_count: 0,
            descriptors: &[],
            prototype: scope.handle(Smi::new(0)),
        })
        .into_global(roots);
    let null = heap
        .allocate::<Object>(ObjectInit {
            map: null_map,
            slots: unsafe { smi_handle::<FixedArray>(roots) },
            elements: unsafe { smi_handle::<Value>(roots) },
            length: 0,
        })
        .into_global(roots);

    // publish the sentinels before anything reads them through `known()`
    // (clear() takes the heap and fetches the hole from there)
    known.the_hole = the_hole;
    known.null = null;
    known.null_map = null_map;
    heap.set_known(known);

    heap.no_gc(|nogc| {
        map_map.heap_ref(nogc).transitions.clear(nogc.heap());
        the_hole_map.heap_ref(nogc).transitions.clear(nogc.heap());
        null_map.heap_ref(nogc).transitions.clear(nogc.heap());
        the_hole_map
            .heap_ref(nogc)
            .prototype
            .set(nogc, the_hole_map, null);
        null_map.heap_ref(nogc).prototype.set(nogc, null_map, null);
    });

    // builtin maps: every allocation init path looks these up, so they must
    // exist before any other object is created (incl. interned strings)
    let smi_map = alloc_map(heap, roots, MapKind::OBJECT);
    let float_map = alloc_map(heap, roots, MapKind::FLOAT);
    let array_map = alloc_map(heap, roots, MapKind::FIXED_ARRAY);
    let byte_array_map = alloc_map(heap, roots, MapKind::FIXED_BYTE_ARRAY);
    let string_map = alloc_map(heap, roots, MapKind::VM_STRING);
    let symbol_map = alloc_map(heap, roots, MapKind::SYMBOL);
    let accessor_pair_map = alloc_map(heap, roots, MapKind::ACCESSOR_PAIR);
    let callable_map = alloc_map(heap, roots, MapKind::CALLABLE_INFO);
    let handler_table_map = alloc_map(heap, roots, MapKind::HANDLER_TABLE);
    let context_map = alloc_map(heap, roots, MapKind::CONTEXT);
    let scope_info_map = alloc_map(heap, roots, MapKind::SCOPE_INFO);

    let function_map = heap
        .allocate::<Map>(MapInit {
            // TODO: arrow/generator functions get a non-constructor map
            // once the compiler distinguishes them
            kind: MapKind::OBJECT
                .union(MapKind::CALLABLE)
                .union(MapKind::CONSTRUCTOR)
                .union(MapKind::EXTENDABLE),
            value_slot_count: 2,
            descriptors: &[],
            prototype: known.null.erase(),
        })
        .into_global(roots);
    let non_constructor_function_map = heap
        .allocate::<Map>(MapInit {
            kind: MapKind::OBJECT
                .union(MapKind::CALLABLE)
                .union(MapKind::EXTENDABLE),
            value_slot_count: 2,
            descriptors: &[],
            prototype: known.null.erase(),
        })
        .into_global(roots);
    let class_constructor_map = heap
        .allocate::<Map>(MapInit {
            kind: MapKind::OBJECT
                .union(MapKind::CALLABLE)
                .union(MapKind::CONSTRUCTOR)
                .union(MapKind::EXTENDABLE)
                .union(MapKind::CLASS_CONSTRUCTOR),
            // slots: [callable info, closure context, instance-field array]
            value_slot_count: 3,
            descriptors: &[],
            prototype: known.null.erase(),
        })
        .into_global(roots);
    known.smi_map = smi_map;
    known.float_map = float_map;
    known.array_map = array_map;
    known.byte_array_map = byte_array_map;
    known.string_map = string_map;
    known.symbol_map = symbol_map;
    known.accessor_pair_map = accessor_pair_map;
    known.callable_map = callable_map;
    known.handler_table_map = handler_table_map;
    known.context_map = context_map;
    known.scope_info_map = scope_info_map;
    known.function_map = function_map;
    known.non_constructor_function_map = non_constructor_function_map;
    known.class_constructor_map = class_constructor_map;
    heap.set_known(known);
}

pub fn intern_well_known_strings(heap: &mut Heap, interner: &StringInterner, roots: &RootHandles) {
    let strings = WellKnownStrings::intern_all(heap, interner, roots);
    let mut known = *heap.known();
    known.strings = strings;
    heap.set_known(known);
}

pub fn bootstrap_well_known(heap: &mut Heap, roots: &RootHandles) {
    let mut known = *heap.known();
    let the_hole = known.the_hole;
    debug_assert!(
        !the_hole.value().is_smi(),
        "bootstrap_basics must run before bootstrap_well_known"
    );

    let data = HandleData::new(the_hole.value());
    let scope = unsafe { HandleScope::from_raw(NonNull::from(&data)) };

    let object_prototype_map = alloc_map(heap, roots, MapKind::OBJECT.union(MapKind::EXTENDABLE));
    let empty_slots = heap.allocate::<FixedArray>(&[]).into_global(roots);
    known.empty_fixed_array = heap.allocate::<FixedArray>(&[]).into_global(roots);
    heap.set_known(known);
    let object_prototype = alloc_object(heap, &scope, roots, object_prototype_map);

    let array_prototype_map = alloc_parent_map(
        heap,
        roots,
        MapKind::ARRAY.union(MapKind::EXTENDABLE),
        object_prototype,
    );
    let array_prototype = alloc_object(heap, &scope, roots, array_prototype_map);
    let js_array_map = alloc_parent_map(
        heap,
        roots,
        MapKind::ARRAY.union(MapKind::EXTENDABLE),
        array_prototype,
    );

    let error_prototype_map = alloc_parent_map(
        heap,
        roots,
        MapKind::OBJECT.union(MapKind::EXTENDABLE),
        object_prototype,
    );
    let error_prototype = alloc_object(heap, &scope, roots, error_prototype_map);

    let error_map = alloc_parent_map(
        heap,
        roots,
        MapKind::OBJECT.union(MapKind::EXTENDABLE),
        error_prototype,
    );

    let object_initial_map = alloc_parent_map(
        heap,
        roots,
        MapKind::OBJECT.union(MapKind::EXTENDABLE),
        object_prototype,
    );
    let global_object = alloc_object(heap, &scope, roots, object_initial_map);

    let undefined_map = alloc_parent_map(heap, roots, MapKind::OBJECT, object_prototype);
    let boolean_map = alloc_parent_map(heap, roots, MapKind::OBJECT, object_prototype);

    let undefined = alloc_object(heap, &scope, roots, undefined_map);
    let false_object = alloc_object(heap, &scope, roots, boolean_map);
    let true_object = alloc_object(heap, &scope, roots, boolean_map);

    let exception_map = alloc_parent_map(heap, roots, MapKind::OBJECT, object_prototype);
    let exception = alloc_object(heap, &scope, roots, exception_map);

    let to_primitive_symbol =
        roots.create_handle(Symbol::new(heap, &scope, b"Symbol.toPrimitive").as_tagged());
    let iterator_symbol =
        roots.create_handle(Symbol::new(heap, &scope, b"Symbol.iterator").as_tagged());

    let empty_scope_info = heap
        .allocate::<ScopeInfo>(ScopeInfoInit { names: empty_slots })
        .into_global(roots);
    let empty_context = heap
        .allocate::<Context>(ContextInit {
            outer: None,
            slots: empty_slots,
            scope_info: empty_scope_info,
        })
        .into_global(roots);

    // %Function.prototype% (ES 19.2.3): the canonical empty function — a
    // callable, non-constructable function whose [[Prototype]] is
    // %Object.prototype%; the [[Prototype]] of ordinary function objects
    // (patched onto function_map at the end). The body is a single
    // `Return` (bytecode::Opcode::Return as u8) so calling it yields
    // undefined.
    let function_prototype_map = alloc_parent_map_with_slots(
        heap,
        roots,
        MapKind::OBJECT
            .union(MapKind::CALLABLE)
            .union(MapKind::EXTENDABLE),
        object_prototype,
        2, // [callable info, context], like function_map
    );
    let empty_code = heap.allocate::<FixedByteArray>(&[1]).into_handle(&scope);
    let empty_info = heap
        .allocate::<CallableInfoObject>(CallableInfoInit {
            bytecode: empty_code,
            constants: known.empty_fixed_array,
            register_count: 0,
            handlers: None,
        })
        .into_handle(&scope);
    let function_prototype = heap
        .new_object(
            &scope,
            function_prototype_map,
            &[empty_info.value(), empty_context.value()],
        )
        .into_global(roots);

    known.undefined = undefined;
    known.undefined_map = undefined_map;
    known.false_object = false_object;
    known.true_object = true_object;
    known.boolean_map = boolean_map;
    known.exception = exception;
    known.to_primitive_symbol = to_primitive_symbol;
    known.iterator_symbol = iterator_symbol;
    known.object_prototype = object_prototype;
    known.array_prototype = array_prototype;
    known.error_prototype = error_prototype;
    known.function_prototype = function_prototype;
    known.error_map = error_map;
    // the per-class error maps and wrapper maps are placeholders until the
    // builtins bootstrap installs their prototypes (they start pointing at
    // the plain error map so no allocation ever reads a garbage map)
    known.type_error_map = error_map;
    known.reference_error_map = error_map;
    known.range_error_map = error_map;
    known.number_wrapper_map = error_map;
    known.boolean_wrapper_map = error_map;
    known.string_wrapper_map = error_map;
    known.exception_map = exception_map;
    known.js_array_map = js_array_map;
    known.empty_context = empty_context;
    known.empty_scope_info = empty_scope_info;
    known.global_object = global_object;
    known.object_initial_map = object_initial_map;
    heap.set_known(known);

    let null = known.null;
    heap.no_gc(|nogc| {
        let o = null.heap_ref(nogc);
        o.slots.set(nogc, null, known.empty_fixed_array);
        o.elements.set(nogc, null, known.empty_fixed_array);
        // ordinary function objects' [[Prototype]] is %Function.prototype%
        // (ES 19.2.3.1): function_map was created with a null placeholder
        // in bootstrap_basics
        known.function_map.heap_ref(nogc).prototype.set(
            nogc,
            known.function_map,
            function_prototype,
        );
        known
            .non_constructor_function_map
            .heap_ref(nogc)
            .prototype
            .set(nogc, known.non_constructor_function_map, function_prototype);
        known.class_constructor_map.heap_ref(nogc).prototype.set(
            nogc,
            known.class_constructor_map,
            function_prototype,
        );
    });
}

pub struct KnownCell {
    known: UnsafeCell<WellKnown>,
}

unsafe impl Send for KnownCell {}
unsafe impl Sync for KnownCell {}

impl KnownCell {
    pub fn new(initial: WellKnown) -> Self {
        Self {
            known: UnsafeCell::new(initial),
        }
    }

    pub fn set(&self, known: WellKnown) {
        unsafe { *self.known.get() = known };
    }

    pub fn get(&self) -> &WellKnown {
        unsafe { &*self.known.get() }
    }
}
