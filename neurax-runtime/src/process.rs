//! Is that process still there?
//!
//! The only question this module answers, and the whole reason `interrupted`
//! can be inferred rather than reported.
//!
//! On Unix, `kill(pid, 0)` sends no signal and returns whether the process
//! exists and could be signalled. That is the standard check, and it has one
//! well-known flaw: pids are reused, so a long-dead run whose pid has since
//! been handed to a text editor would read as alive.
//!
//! On Linux that is closed properly, by comparing the process's own start
//! time from `/proc/<pid>/stat` against the run's. A reused pid belongs to a
//! process that started later than the run did, and the difference is
//! decisive. Where `/proc` is not available the plain existence check stands,
//! which is the honest ceiling for that platform rather than a pretence.

/// True when a process with this pid exists and we could signal it.
pub fn is_running(pid: u32) -> bool {
    #[cfg(unix)]
    {
        if pid == 0 {
            return false;
        }
        // SAFETY: `kill` with signal 0 performs error checking only — it
        // delivers nothing and cannot affect the target process.
        let alive = unsafe { libc::kill(pid as libc::pid_t, 0) } == 0;
        if alive {
            return true;
        }
        // EPERM means the process exists but belongs to another user. That is
        // still "running" for our purposes, and treating it as gone would
        // report a live run as interrupted.
        std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}

/// The process's start time, in clock ticks since boot, from `/proc`.
///
/// Used to tell a live run from a pid that has been recycled. `None` when the
/// process is gone or `/proc` is unavailable, and callers treat that as "no
/// evidence" rather than as proof of either state.
#[cfg(target_os = "linux")]
pub fn start_ticks(pid: u32) -> Option<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // Field 22 is starttime, but fields 1 and 2 are the pid and the executable
    // name — and the name is parenthesised and may itself contain spaces and
    // parentheses. Splitting on whitespace from the left is the classic bug
    // here; everything after the final ')' is the part that can be split
    // safely.
    let rest = &stat[stat.rfind(')')? + 1..];
    rest.split_whitespace().nth(19)?.parse().ok()
}

#[cfg(not(target_os = "linux"))]
pub fn start_ticks(_pid: u32) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn this_process_is_running() {
        assert!(is_running(std::process::id()));
    }

    #[test]
    fn an_impossible_pid_is_not_running() {
        // Above any pid_max a Linux system will hand out.
        assert!(!is_running(u32::MAX - 1));
    }

    #[test]
    fn pid_zero_is_never_a_run() {
        // `kill(0, 0)` signals the caller's whole process group, which would
        // report "alive" for a state file that simply never recorded a pid.
        assert!(!is_running(0));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn start_time_is_readable_for_a_live_process() {
        assert!(start_ticks(std::process::id()).is_some());
        assert!(start_ticks(u32::MAX - 1).is_none());
    }
}
