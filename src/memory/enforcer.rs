// Process-level memory enforcer
// Monitors total memory usage and evicts models when limits are exceeded

pub struct MemoryEnforcer {
    max_process_memory: u64,
}

impl MemoryEnforcer {
    pub fn new(max_bytes: u64) -> Self {
        Self { max_process_memory: max_bytes }
    }

    pub fn current_usage_bytes(&self) -> u64 {
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            let pid = std::process::id();
            if let Ok(out) = Command::new("ps").args(["-o", "rss=", "-p", &pid.to_string()]).output() {
                if let Ok(s) = String::from_utf8(out.stdout) {
                    return s.trim().parse::<u64>().unwrap_or(0) * 1024;
                }
            }
        }
        0
    }

    pub fn should_evict(&self) -> bool {
        self.current_usage_bytes() > self.max_process_memory
    }
}
