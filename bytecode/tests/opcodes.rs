use bytecode::Opcode;

#[test]
fn from_byte_roundtrips_every_opcode() {
    for byte in 0..=u8::MAX {
        if let Some(op) = Opcode::from_byte(byte) {
            assert_eq!(
                op as u8, byte,
                "from_byte({byte}) resolved to an opcode with a different discriminant"
            );
        }
    }
}
