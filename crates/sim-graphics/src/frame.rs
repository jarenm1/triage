use crate::{Camera, RenderPrimitive};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewId(u32);

impl ViewId {
    pub(crate) fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("frame exhausted u32 view identifiers"))
    }

    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewKey(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewKind {
    Display,
    Sensor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewOutputs(u8);

impl ViewOutputs {
    pub const COLOR: Self = Self(1 << 0);
    pub const DEPTH: Self = Self(1 << 1);
    pub const OBJECT_ID: Self = Self(1 << 2);

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub(crate) const fn bits(self) -> u8 {
        self.0
    }
}

impl std::ops::BitOr for ViewOutputs {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for ViewOutputs {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RenderView {
    pub key: ViewKey,
    pub kind: ViewKind,
    pub camera: Camera,
    pub width: u32,
    pub height: u32,
    pub outputs: ViewOutputs,
}

pub struct Frame {
    primitives: Vec<RenderPrimitive>,
    views: Vec<RenderView>,
}

impl Frame {
    pub fn new() -> Self {
        Self {
            primitives: Vec::new(),
            views: Vec::new(),
        }
    }

    pub fn with_capacity(primitive_capacity: usize, view_capacity: usize) -> Self {
        Self {
            primitives: Vec::with_capacity(primitive_capacity),
            views: Vec::with_capacity(view_capacity),
        }
    }

    pub fn begin(&mut self) -> &mut Self {
        self.primitives.clear();
        self.views.clear();
        self
    }

    pub fn draw(&mut self, primitive: RenderPrimitive) -> &mut Self {
        self.primitives.push(primitive);
        self
    }

    pub fn add_view(&mut self, view: RenderView) -> ViewId {
        assert!(view.width > 0 && view.height > 0, "view dimensions must be non-zero");
        let index = u32::try_from(self.views.len()).expect("frame exhausted u32 view identifiers");
        self.views.push(view);
        ViewId(index)
    }

    pub fn primitives(&self) -> &[RenderPrimitive] {
        &self.primitives
    }

    pub fn views(&self) -> &[RenderView] {
        &self.views
    }

    pub fn view(&self, id: ViewId) -> Option<&RenderView> {
        self.views.get(id.index())
    }

    pub fn primitive_capacity(&self) -> usize {
        self.primitives.capacity()
    }

    pub fn first_view(&self, kind: ViewKind) -> Option<ViewId> {
        self.views
            .iter()
            .position(|view| view.kind == kind)
            .map(ViewId::from_index)
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self::new()
    }
}
