//! one forward pass, whichever engine runs it

use crate::artefacts::Artefacts;
use crate::batch::HostBatch;
use crate::cpu::CpuEngine;
use crate::error::Result;
use crate::outputs::Outputs;

pub trait Forward {
    fn artefacts(&self) -> &Artefacts;
    /// `graph`: an engine that can replay a prepared plan for the batch's shape may; the CPU engine
    /// has one path
    fn run(&self, b: &HostBatch, graph: bool) -> Result<Outputs>;
}

impl Forward for CpuEngine {
    fn artefacts(&self) -> &Artefacts {
        &self.a
    }
    fn run(&self, b: &HostBatch, _graph: bool) -> Result<Outputs> {
        self.forward(b)
    }
}
