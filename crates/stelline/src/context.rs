use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ContextMode {
    Full,   // Overwrite entire file
    Append, // Append new lines
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextTarget {
    pub name: String,
    pub path: String,
    pub instruction: String,
    pub enabled: bool,
    pub mode: ContextMode,
}

pub struct ContextManager {
    targets: Vec<ContextTarget>,
}

impl ContextManager {
    pub fn new() -> Self {
        Self {
            targets: Vec::new(),
        }
    }

    pub fn add_target(&mut self, target: ContextTarget) {
        self.targets.push(target);
    }

    pub fn remove_target(&mut self, name: &str) -> bool {
        let len = self.targets.len();
        self.targets.retain(|t| t.name != name);
        self.targets.len() < len
    }

    pub fn list_targets(&self) -> &[ContextTarget] {
        &self.targets
    }

    pub fn load_context(&self, name: &str) -> crate::error::Result<String> {
        let target = self
            .targets
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| {
                crate::error::StellineError::Other(format!("Context target not found: {name}"))
            })?;
        std::fs::read_to_string(&target.path).map_err(crate::error::StellineError::Io)
    }

    pub fn apply_update(&self, name: &str, content: &str) -> crate::error::Result<()> {
        let target = self
            .targets
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| {
                crate::error::StellineError::Other(format!("Context target not found: {name}"))
            })?;

        match target.mode {
            ContextMode::Full => {
                std::fs::write(&target.path, content)?;
            }
            ContextMode::Append => {
                use std::io::Write;
                let mut f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&target.path)?;
                writeln!(f, "{content}")?;
            }
        }
        Ok(())
    }
}

impl Default for ContextManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_target(name: &str) -> ContextTarget {
        ContextTarget {
            name: name.to_string(),
            path: format!("/tmp/{name}.txt"),
            instruction: "keep it updated".to_string(),
            enabled: true,
            mode: ContextMode::Full,
        }
    }

    #[test]
    fn add_and_list() {
        let mut mgr = ContextManager::new();
        assert!(mgr.list_targets().is_empty());

        mgr.add_target(make_target("alpha"));
        mgr.add_target(make_target("beta"));

        let names: Vec<&str> = mgr.list_targets().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn remove_existing() {
        let mut mgr = ContextManager::new();
        mgr.add_target(make_target("alpha"));
        mgr.add_target(make_target("beta"));

        let removed = mgr.remove_target("alpha");
        assert!(removed);
        assert_eq!(mgr.list_targets().len(), 1);
        assert_eq!(mgr.list_targets()[0].name, "beta");
    }

    #[test]
    fn remove_nonexistent() {
        let mut mgr = ContextManager::new();
        mgr.add_target(make_target("alpha"));

        let removed = mgr.remove_target("ghost");
        assert!(!removed);
        assert_eq!(mgr.list_targets().len(), 1);
    }

    #[test]
    fn list_is_empty_after_removing_all() {
        let mut mgr = ContextManager::new();
        mgr.add_target(make_target("only"));
        mgr.remove_target("only");
        assert!(mgr.list_targets().is_empty());
    }
}
