//! Forward/backward label patching for the bytecode stream.
//!
//! A label is a not-yet-known pc. `bind` fixes it, `emit_jump` reserves a
//! forced-wide jump (always 2-byte offset) that `patch_all` rewrites once
//! the target is known.

#[derive(Default)]
pub struct Label {
    pos: Option<usize>,
    patches: Vec<usize>,
}

impl Label {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fix the label at the current pc.
    pub fn bind(&mut self, code: &[u8]) {
        debug_assert!(self.pos.is_none(), "label bound twice");
        self.pos = Some(code.len());
    }

    /// Fix the label at an explicit pc (backward branches).
    pub fn bind_at(&mut self, pc: usize) {
        debug_assert!(self.pos.is_none(), "label bound twice");
        self.pos = Some(pc);
    }

    /// Target pc: only valid after `bind`.
    pub fn pos(&self) -> usize {
        self.pos.expect("label used before bind")
    }

    /// Record a jump at `jump_pc` (the pc of the jump instruction itself)
    /// that must land here; patched by [`patch_all`](Self::patch_all).
    pub fn patch_here(&mut self, jump_pc: usize) {
        self.patches.push(jump_pc);
    }

    /// Rewrite every recorded relative jump to target this label.
    pub fn patch_all(&self, code: &mut [u8]) {
        let target = self.pos();
        for &jump_pc in &self.patches {
            patch_jump(code, jump_pc, target);
        }
    }
}

/// Rewrite a relative jump operand at `jump_pc` to target `target_pc`.
/// Jumps are always emitted forced-wide: `[Wide][op][offset:2]`.
pub fn patch_jump(code: &mut [u8], jump_pc: usize, target_pc: usize) {
    debug_assert_eq!(
        code[jump_pc],
        bytecode::Opcode::Wide as u8,
        "jump must be wide"
    );
    let offset = target_pc as i64 - jump_pc as i64;
    let offset = i32::try_from(offset).expect("jump offset out of range");
    let bytes = offset.to_le_bytes();
    // Wide prefix (1) + opcode (1), then the 2-byte immediate
    code[jump_pc + 2..jump_pc + 4].copy_from_slice(&bytes[..2]);
}
