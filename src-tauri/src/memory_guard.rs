//! A memory ceiling over the backend and every process it starts.
//!
//! Past `SOFT_LIMIT_BYTES` for the whole tree the guard logs which processes
//! held what, ends the tree and leaves the restart to the watchdog. Every
//! backend instance has a generation, and the guard's verdict names the one it
//! ended. Windows counts commit in a Job Object per instance, which also carries
//! `HARD_LIMIT_BYTES`; macOS polls each process's physical footprint every
//! `POLL_INTERVAL`. Nothing reaches the user.

use std::sync::Mutex;
use std::time::{Duration, Instant};

const GIB: u64 = 1 << 30;

/// The whole-tree size past which the backend is ended and restarted.
pub const SOFT_LIMIT_BYTES: u64 = 8 * GIB;

/// The Windows job's commit ceiling: an allocation past it fails.
#[cfg(windows)]
pub const HARD_LIMIT_BYTES: u64 = 10 * GIB;

/// How often macOS samples the tree's footprint.
#[cfg(target_os = "macos")]
const POLL_INTERVAL: Duration = Duration::from_secs(3);

/// A second signal this soon after the crossing last acted on reports that same crossing.
const DUPLICATE_SIGNAL_WINDOW: Duration = Duration::from_secs(1);

/// A crossing this soon after the previous one lengthens the wait before the next restart.
const REPEAT_WINDOW: Duration = Duration::from_secs(10 * 60);

const REPEAT_RESTART_DELAY: Duration = Duration::from_secs(30);
const MAX_RESTART_DELAY: Duration = Duration::from_secs(10 * 60);

/// Which ceiling the tree crossed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MemoryLimit {
    Soft,
    #[cfg(windows)]
    Hard,
}

/// One process of the tree and the memory it holds.
pub struct ProcessMemory {
    pub pid: u32,
    pub name: String,
    pub bytes: u64,
}

/// Memory restarts so far: when the last one was, and how many came in a row.
struct RestartBackoff {
    last: Option<Instant>,
    streak: u32,
}

impl RestartBackoff {
    /// The extra wait before restarting after a crossing at `now`, or `None`
    /// when `now` repeats the crossing just acted on.
    fn record(&mut self, now: Instant) -> Option<Duration> {
        if let Some(last) = self.last {
            let since = now.saturating_duration_since(last);
            if since < DUPLICATE_SIGNAL_WINDOW {
                return None;
            }
            self.streak = if since < REPEAT_WINDOW { self.streak + 1 } else { 0 };
        }
        self.last = Some(now);
        Some(restart_delay(self.streak))
    }
}

fn restart_delay(streak: u32) -> Duration {
    if streak == 0 {
        return Duration::ZERO;
    }
    REPEAT_RESTART_DELAY
        .saturating_mul(2u32.saturating_pow(streak - 1))
        .min(MAX_RESTART_DELAY)
}

/// The backend generation the guard last ended and the wait it set before its restart.
struct RestartMarker(Mutex<Option<(u64, Duration)>>);

impl RestartMarker {
    fn set(&self, generation: u64, delay: Duration) {
        if let Ok(mut pending) = self.0.lock() {
            *pending = Some((generation, delay));
        }
    }

    /// Whether the guard ended `generation`; reading leaves the marker in place.
    fn ended(&self, generation: u64) -> bool {
        self.0.lock().ok().is_some_and(|pending| matches!(*pending, Some((ended, _)) if ended == generation))
    }

    /// The wait owed before restarting after `generation`, once; a marker for
    /// any other generation is dropped and owes nothing.
    fn take_for(&self, generation: u64) -> Option<Duration> {
        let taken = self.0.lock().ok().and_then(|mut pending| pending.take())?;
        (taken.0 == generation).then_some(taken.1)
    }
}

static BACKOFF: Mutex<RestartBackoff> = Mutex::new(RestartBackoff { last: None, streak: 0 });
static PENDING_RESTART: RestartMarker = RestartMarker(Mutex::new(None));

/// Whether backend `generation` was ended by the guard rather than failing:
/// such an end is neither reported nor counted toward giving up.
pub fn ended_by_guard(generation: u64) -> bool {
    PENDING_RESTART.ended(generation)
}

/// The extra wait before the backend that follows `generation` starts, owed
/// once when the guard ended `generation`; zero otherwise.
pub fn wait_owed_after(generation: u64) -> Duration {
    PENDING_RESTART.take_for(generation).unwrap_or(Duration::ZERO)
}

#[cfg(any(target_os = "macos", test))]
fn tree_total(processes: &[ProcessMemory]) -> u64 {
    processes.iter().map(|process| process.bytes).sum()
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB as f64)
}

/// The log line for a crossing: the limit, the size reached and each process.
fn describe(limit: MemoryLimit, reached: u64, processes: &[ProcessMemory]) -> String {
    let listed: Vec<String> = processes
        .iter()
        .map(|process| format!("{} {} {}", process.pid, process.name, gib(process.bytes)))
        .collect();
    format!(
        "Memory guard: the backend's processes crossed the {:?} limit, reaching {}: {}",
        limit,
        gib(reached),
        listed.join(", ")
    )
}

/// Log `report`, mark `generation` as ended by the guard and run `end_tree`,
/// unless the crossing repeats one already acted on.
fn act_on_crossing(generation: u64, report: &str, end_tree: impl FnOnce()) {
    let delay = BACKOFF.lock().ok().and_then(|mut backoff| backoff.record(Instant::now()));
    let Some(delay) = delay else {
        log::info!("{}; already ending them", report);
        return;
    };
    log::warn!("{}; ending them and restarting the backend after {:?} more", report, delay);
    PENDING_RESTART.set(generation, delay);
    end_tree();
}

#[cfg(windows)]
pub use self::windows_job::instance_job;

#[cfg(all(windows, test))]
pub(crate) use self::windows_job::create_job;

#[cfg(windows)]
mod windows_job {
    use super::{act_on_crossing, describe, MemoryLimit, ProcessMemory, SOFT_LIMIT_BYTES};
    use core::ffi::c_void;
    use std::sync::{Mutex, OnceLock, PoisonError};
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::JobObjects::{
        CreateJobObjectW, JobObjectAssociateCompletionPortInformation, JobObjectBasicProcessIdList,
        JobObjectExtendedLimitInformation, JobObjectLimitViolationInformation,
        JobObjectNotificationLimitInformation, QueryInformationJobObject,
        SetInformationJobObject, TerminateJobObject, JOBOBJECTINFOCLASS,
        JOBOBJECT_ASSOCIATE_COMPLETION_PORT, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOBOBJECT_LIMIT_VIOLATION_INFORMATION, JOBOBJECT_NOTIFICATION_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows_sys::Win32::System::IO::{CreateIoCompletionPort, GetQueuedCompletionStatus};
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, INFINITE, PROCESS_NAME_WIN32,
        PROCESS_QUERY_LIMITED_INFORMATION,
    };

    const JOB_OBJECT_MSG_JOB_MEMORY_LIMIT: u32 = 10;
    const JOB_OBJECT_MSG_NOTIFICATION_LIMIT: u32 = 11;
    const LISTED_PROCESSES_MAX: usize = 64;

    #[repr(C)]
    struct ProcessIdList {
        assigned: u32,
        listed: u32,
        ids: [usize; LISTED_PROCESSES_MAX],
    }

    static PORT: OnceLock<Option<isize>> = OnceLock::new();
    static LIVE_JOB: Mutex<Option<(u64, InstanceJob)>> = Mutex::new(None);

    /// A job handle owned by one backend instance; dropping it ends whatever
    /// of that instance's tree still runs.
    struct InstanceJob(isize);

    impl Drop for InstanceJob {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.0 as HANDLE);
            }
        }
    }

    /// A kill-on-close job for backend `generation` with the hard limit, and the
    /// soft limit unless arming it fails. Only the shell holds it, so the tree
    /// dies with the shell however it dies (ADR 023); it replaces, and so ends,
    /// the previous instance's job. `None` when no job can be made.
    pub fn instance_job(generation: u64) -> Option<isize> {
        let job = create_job(super::HARD_LIMIT_BYTES)?;
        let armed = watch_port().is_some_and(|port| arm_job(job, port, generation, SOFT_LIMIT_BYTES));
        if !armed {
            log::warn!("Memory guard: backend {} has no soft memory limit", generation);
        }
        let mut live = LIVE_JOB.lock().unwrap_or_else(PoisonError::into_inner);
        *live = Some((generation, InstanceJob(job)));
        Some(job)
    }

    /// A job that ends its processes when its last handle closes and refuses
    /// commit past `hard_limit` bytes.
    pub(crate) fn create_job(hard_limit: u64) -> Option<isize> {
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                log::warn!("CreateJobObjectW failed; force-kill orphan protection unavailable for this backend");
                return None;
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags =
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_JOB_MEMORY;
            info.JobMemoryLimit = hard_limit as usize;
            if !set(job as isize, JobObjectExtendedLimitInformation, &info) {
                log::warn!("SetInformationJobObject failed; force-kill orphan protection unavailable for this backend");
                CloseHandle(job);
                return None;
            }
            Some(job as isize)
        }
    }

    /// The one port every instance job reports to, with the thread that acts on it.
    fn watch_port() -> Option<isize> {
        *PORT.get_or_init(|| {
            let port = new_port()?;
            let started = std::thread::Builder::new()
                .name("memory-guard".to_string())
                .spawn(move || {
                    while let Some((limit, generation)) = next_limit_crossing(port, INFINITE) {
                        let live = LIVE_JOB.lock().unwrap_or_else(PoisonError::into_inner);
                        let owner = live.as_ref().map(|(owner, job)| (*owner, job.0));
                        let Some(job) = job_to_end(owner, generation) else {
                            log::info!(
                                "Memory guard: ended backend {} crossed the {:?} limit; nothing to end",
                                generation,
                                limit
                            );
                            continue;
                        };
                        let report = crossing_report(job, limit);
                        act_on_crossing(generation, &report, || unsafe {
                            TerminateJobObject(job as HANDLE, 1);
                        });
                    }
                    log::warn!("Memory guard: the completion port failed; no memory restart from now on");
                });
            if started.is_err() {
                log::warn!("Memory guard: its thread could not start; no memory restart this session");
                return None;
            }
            Some(port)
        })
    }

    /// The job to end for a crossing reported by `generation`: the live one,
    /// and only while it still belongs to `generation`.
    pub(super) fn job_to_end(live: Option<(u64, isize)>, generation: u64) -> Option<isize> {
        live.filter(|(owner, _)| *owner == generation).map(|(_, job)| job)
    }

    pub(super) fn new_port() -> Option<isize> {
        let port = unsafe { CreateIoCompletionPort(INVALID_HANDLE_VALUE, std::ptr::null_mut(), 0, 1) };
        (!port.is_null()).then_some(port as isize)
    }

    /// Have `job` report to `port` under `generation` once its commit passes `soft_limit` bytes.
    pub(super) fn arm_job(job: isize, port: isize, generation: u64, soft_limit: u64) -> bool {
        let link = JOBOBJECT_ASSOCIATE_COMPLETION_PORT {
            CompletionKey: generation as usize as *mut c_void,
            CompletionPort: port as HANDLE,
        };
        let mut notify: JOBOBJECT_NOTIFICATION_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        notify.JobMemoryLimit = soft_limit;
        notify.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY;
        set(job, JobObjectAssociateCompletionPortInformation, &link)
            && set(job, JobObjectNotificationLimitInformation, &notify)
    }

    /// Wait up to `timeout_ms` for the next memory-limit message on `port` and
    /// the generation whose job sent it, skipping every other job message;
    /// `None` on a timeout or a failed port.
    pub(super) fn next_limit_crossing(port: isize, timeout_ms: u32) -> Option<(MemoryLimit, u64)> {
        loop {
            let mut message = 0u32;
            let mut key = 0usize;
            let mut overlapped = std::ptr::null_mut();
            let received = unsafe {
                GetQueuedCompletionStatus(port as HANDLE, &mut message, &mut key, &mut overlapped, timeout_ms)
            };
            if received == 0 {
                return None;
            }
            match message {
                JOB_OBJECT_MSG_NOTIFICATION_LIMIT => return Some((MemoryLimit::Soft, key as u64)),
                JOB_OBJECT_MSG_JOB_MEMORY_LIMIT => return Some((MemoryLimit::Hard, key as u64)),
                _ => {}
            }
        }
    }

    /// Describe the crossing. Reading the violation re-arms the soft limit,
    /// which reports nothing more until it is read.
    pub(super) fn crossing_report(job: isize, limit: MemoryLimit) -> String {
        query::<JOBOBJECT_LIMIT_VIOLATION_INFORMATION>(job, JobObjectLimitViolationInformation);
        let peak = query::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>(job, JobObjectExtendedLimitInformation)
            .map_or(0, |info| info.PeakJobMemoryUsed as u64);
        describe(limit, peak, &job_processes(job))
    }

    fn job_processes(job: isize) -> Vec<ProcessMemory> {
        let mut list: ProcessIdList = unsafe { std::mem::zeroed() };
        unsafe {
            QueryInformationJobObject(
                job as HANDLE,
                JobObjectBasicProcessIdList,
                &mut list as *mut _ as *mut c_void,
                std::mem::size_of::<ProcessIdList>() as u32,
                std::ptr::null_mut(),
            );
        }
        let listed = (list.listed as usize).min(LISTED_PROCESSES_MAX);
        list.ids[..listed].iter().map(|&pid| process_memory(pid as u32)).collect()
    }

    fn process_memory(pid: u32) -> ProcessMemory {
        let mut memory = ProcessMemory { pid, name: "?".to_string(), bytes: 0 };
        unsafe {
            let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if process.is_null() {
                return memory;
            }
            let mut counters: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
            let size = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
            if K32GetProcessMemoryInfo(process, &mut counters, size) != 0 {
                memory.bytes = counters.PagefileUsage as u64;
            }
            let mut path = [0u16; 1024];
            let mut length = path.len() as u32;
            if QueryFullProcessImageNameW(process, PROCESS_NAME_WIN32, path.as_mut_ptr(), &mut length) != 0 {
                let full = String::from_utf16_lossy(&path[..length as usize]);
                memory.name = full.rsplit('\\').next().unwrap_or(&full).to_string();
            }
            CloseHandle(process);
        }
        memory
    }

    fn set<T>(job: isize, class: JOBOBJECTINFOCLASS, info: &T) -> bool {
        let size = std::mem::size_of::<T>() as u32;
        unsafe { SetInformationJobObject(job as HANDLE, class, info as *const T as *const c_void, size) != 0 }
    }

    fn query<T>(job: isize, class: JOBOBJECTINFOCLASS) -> Option<T> {
        let mut info: T = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<T>() as u32;
        let read = unsafe {
            QueryInformationJobObject(job as HANDLE, class, &mut info as *mut T as *mut c_void, size, std::ptr::null_mut())
        };
        (read != 0).then_some(info)
    }
}

#[cfg(target_os = "macos")]
pub use self::macos_tree::{end_process_group, reap_orphaned_engines, watch_backend_tree};

/// The local engine's executable name, which a leftover of ours carries.
#[cfg(any(target_os = "macos", test))]
const ENGINE_EXECUTABLE_NAME: &str = "whisper-server";

/// Whether a process at `path` whose parent is `parent` is an engine of ours
/// left running after its backend died: adopted by launchd, from one of `roots`.
#[cfg(any(target_os = "macos", test))]
fn is_orphaned_engine(path: &std::path::Path, parent: u32, roots: &[std::path::PathBuf]) -> bool {
    parent == 1
        && path.file_name().is_some_and(|name| name == ENGINE_EXECUTABLE_NAME)
        && roots.iter().any(|root| path.starts_with(root))
}

#[cfg(target_os = "macos")]
mod macos_tree {
    use super::{
        act_on_crossing, describe, is_orphaned_engine, tree_total, MemoryLimit, ProcessMemory,
        POLL_INTERVAL, SOFT_LIMIT_BYTES,
    };
    use core::ffi::c_void;
    use std::path::PathBuf;

    const LISTED_PROCESSES_MAX: usize = 4096;

    /// Start the thread that sums the backend tree's footprint every
    /// `POLL_INTERVAL` and ends the tree past `SOFT_LIMIT_BYTES`.
    pub fn watch_backend_tree() {
        let started = std::thread::Builder::new()
            .name("memory-guard".to_string())
            .spawn(|| loop {
                std::thread::sleep(POLL_INTERVAL);
                let Some((root, generation)) = crate::backend::backend_tree_root() else {
                    continue;
                };
                let pids = tree(root as i32);
                let processes: Vec<ProcessMemory> = pids
                    .iter()
                    .map(|&pid| ProcessMemory { pid: pid as u32, name: String::new(), bytes: footprint(pid) })
                    .collect();
                let total = tree_total(&processes);
                if total <= SOFT_LIMIT_BYTES {
                    continue;
                }
                let named: Vec<ProcessMemory> = processes
                    .into_iter()
                    .map(|process| ProcessMemory { name: name(process.pid as i32), ..process })
                    .collect();
                let report = describe(MemoryLimit::Soft, total, &named);
                act_on_crossing(generation, &report, || end_tree(root as i32, &pids));
            });
        if started.is_err() {
            log::warn!("Memory guard: its thread could not start; no memory restart this session");
        }
    }

    /// `SIGKILL` the process group `leader` led, reaching what it started even
    /// after `leader` itself is gone. No group by that id means nothing to do.
    pub fn end_process_group(leader: u32) {
        unsafe {
            libc::killpg(leader as libc::pid_t, libc::SIGKILL);
        }
    }

    /// `SIGKILL` every engine of ours that launchd adopted from a dead backend.
    pub fn reap_orphaned_engines(roots: &[PathBuf]) {
        for pid in all_pids() {
            let (Some(path), Some(parent)) = (path(pid), parent(pid)) else {
                continue;
            };
            if is_orphaned_engine(&path, parent, roots) {
                log::warn!("Found orphan {:?} (PID {}) — reaping before spawn", path, pid);
                unsafe {
                    libc::kill(pid, libc::SIGKILL);
                }
            }
        }
    }

    fn end_tree(root: i32, pids: &[i32]) {
        unsafe {
            if libc::getpgid(root) == root {
                libc::killpg(root, libc::SIGKILL);
            } else {
                for &pid in pids {
                    libc::kill(pid, libc::SIGKILL);
                }
            }
        }
    }

    /// `root` and every descendant of it.
    pub(super) fn tree(root: i32) -> Vec<i32> {
        let mut pids = vec![root];
        let mut next = 0;
        while next < pids.len() && pids.len() < LISTED_PROCESSES_MAX {
            for pid in children(pids[next]) {
                if !pids.contains(&pid) {
                    pids.push(pid);
                }
            }
            next += 1;
        }
        pids
    }

    fn children(pid: i32) -> Vec<i32> {
        let mut buffer = [0 as libc::pid_t; 256];
        let size = std::mem::size_of_val(&buffer) as libc::c_int;
        let count = unsafe { libc::proc_listchildpids(pid, buffer.as_mut_ptr() as *mut c_void, size) };
        listed(&buffer, count)
    }

    fn all_pids() -> Vec<i32> {
        let mut buffer = vec![0 as libc::pid_t; LISTED_PROCESSES_MAX];
        let size = (buffer.len() * std::mem::size_of::<libc::pid_t>()) as libc::c_int;
        let count = unsafe { libc::proc_listallpids(buffer.as_mut_ptr() as *mut c_void, size) };
        listed(&buffer, count)
    }

    fn listed(buffer: &[libc::pid_t], count: libc::c_int) -> Vec<i32> {
        let count = (count.max(0) as usize).min(buffer.len());
        buffer[..count].iter().copied().filter(|&pid| pid > 0).collect()
    }

    /// The physical footprint Activity Monitor shows as Memory; 0 when unreadable.
    pub(super) fn footprint(pid: i32) -> u64 {
        let mut info: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
        let read = unsafe {
            libc::proc_pid_rusage(pid, libc::RUSAGE_INFO_V4, &mut info as *mut _ as *mut libc::rusage_info_t)
        };
        if read == 0 {
            info.ri_phys_footprint
        } else {
            0
        }
    }

    fn name(pid: i32) -> String {
        let mut buffer = [0u8; 256];
        let length = unsafe { libc::proc_name(pid, buffer.as_mut_ptr() as *mut c_void, buffer.len() as u32) };
        String::from_utf8_lossy(&buffer[..length.max(0) as usize]).to_string()
    }

    fn path(pid: i32) -> Option<PathBuf> {
        let mut buffer = vec![0u8; libc::PROC_PIDPATHINFO_MAXSIZE as usize];
        let length = unsafe { libc::proc_pidpath(pid, buffer.as_mut_ptr() as *mut c_void, buffer.len() as u32) };
        (length > 0).then(|| PathBuf::from(String::from_utf8_lossy(&buffer[..length as usize]).to_string()))
    }

    fn parent(pid: i32) -> Option<u32> {
        let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
        let read = unsafe {
            libc::proc_pidinfo(pid, libc::PROC_PIDTBSDINFO, 0, &mut info as *mut _ as *mut c_void, size)
        };
        (read == size).then_some(info.pbi_ppid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn fresh_backoff() -> RestartBackoff {
        RestartBackoff { last: None, streak: 0 }
    }

    fn process(pid: u32, name: &str, bytes: u64) -> ProcessMemory {
        ProcessMemory { pid, name: name.to_string(), bytes }
    }

    #[test]
    fn a_first_memory_restart_adds_no_wait_to_the_watchdogs_own() {
        assert_eq!(fresh_backoff().record(Instant::now()), Some(Duration::ZERO));
    }

    #[test]
    fn crossings_in_a_row_double_the_wait_up_to_its_ceiling() {
        let mut backoff = fresh_backoff();
        let start = Instant::now();
        let waits: Vec<Option<Duration>> = (0..7u64)
            .map(|minute| backoff.record(start + Duration::from_secs(60 * minute)))
            .collect();
        let seconds = |s: u64| Some(Duration::from_secs(s));
        assert_eq!(
            waits,
            vec![seconds(0), seconds(30), seconds(60), seconds(120), seconds(240), seconds(480), seconds(600)]
        );
    }

    #[test]
    fn a_crossing_long_after_the_last_starts_the_wait_over() {
        let mut backoff = fresh_backoff();
        let start = Instant::now();
        backoff.record(start);
        backoff.record(start + Duration::from_secs(60));
        let later = start + Duration::from_secs(60) + REPEAT_WINDOW;
        assert_eq!(backoff.record(later), Some(Duration::ZERO));
    }

    #[test]
    fn a_second_signal_for_the_same_crossing_ends_nothing_more() {
        let mut backoff = fresh_backoff();
        let start = Instant::now();
        backoff.record(start);
        assert_eq!(backoff.record(start + Duration::from_millis(5)), None);
        assert_eq!(backoff.record(start + Duration::from_secs(60)), Some(REPEAT_RESTART_DELAY));
    }

    fn marker_for(generation: u64, delay: Duration) -> RestartMarker {
        let marker = RestartMarker(Mutex::new(None));
        marker.set(generation, delay);
        marker
    }

    #[test]
    fn the_generation_the_guard_ended_is_not_a_failure_and_owes_its_wait_once() {
        let marker = marker_for(7, REPEAT_RESTART_DELAY);
        assert!(marker.ended(7), "the end the guard caused must not be counted or reported");
        assert_eq!(marker.take_for(7), Some(REPEAT_RESTART_DELAY));
        assert_eq!(marker.take_for(7), None, "the wait is owed once");
    }

    #[test]
    fn a_marker_for_an_earlier_generation_leaves_the_next_failure_a_failure() {
        let marker = marker_for(7, REPEAT_RESTART_DELAY);
        assert!(!marker.ended(8), "generation 8 failing on its own must be counted and reported");
        assert_eq!(marker.take_for(8), None, "generation 8 owes no wait for 7's crossing");
        assert!(!marker.ended(7), "the stale marker is gone once read");
    }

    #[test]
    fn the_tree_total_is_every_process_summed() {
        let processes = [process(1, "backend", 3 * GIB), process(2, "whisper-server", 5 * GIB + 1)];
        let total = tree_total(&processes);
        assert_eq!(total, 8 * GIB + 1);
        assert!(total > SOFT_LIMIT_BYTES);
        assert!(tree_total(&processes[..1]) <= SOFT_LIMIT_BYTES);
    }

    #[test]
    fn the_log_line_names_the_limit_the_size_and_every_process() {
        let line = describe(
            MemoryLimit::Soft,
            8 * GIB + GIB / 2,
            &[process(41, "justsay-backend", GIB), process(42, "whisper-server", 7 * GIB)],
        );
        assert!(line.contains("Soft limit, reaching 8.5 GiB"), "{}", line);
        assert!(line.contains("41 justsay-backend 1.0 GiB, 42 whisper-server 7.0 GiB"), "{}", line);
    }

    #[test]
    fn only_an_engine_of_ours_adopted_by_launchd_is_an_orphan() {
        let roots = [PathBuf::from("/Applications/JustSay.app/Contents/Resources")];
        let ours = Path::new("/Applications/JustSay.app/Contents/Resources/whisper-cpp-metal/whisper-server");
        assert!(is_orphaned_engine(ours, 1, &roots));
        assert!(!is_orphaned_engine(ours, 4242, &roots), "a live backend's engine is not an orphan");
        assert!(!is_orphaned_engine(Path::new("/opt/homebrew/bin/whisper-server"), 1, &roots));
        let other = Path::new("/Applications/JustSay.app/Contents/Resources/justsay-audiotap");
        assert!(!is_orphaned_engine(other, 1, &roots));
    }

    #[cfg(windows)]
    #[test]
    fn a_crossing_of_an_ended_instance_never_ends_the_live_one() {
        let live = Some((8, 0x80));
        assert_eq!(windows_job::job_to_end(live, 7), None);
        assert_eq!(windows_job::job_to_end(live, 8), Some(0x80));
        assert_eq!(windows_job::job_to_end(None, 8), None);
    }

    #[cfg(windows)]
    #[test]
    fn a_child_past_the_soft_limit_is_reported_on_the_jobs_port_under_its_generation() {
        use std::os::windows::io::AsRawHandle;
        use std::os::windows::process::CommandExt;
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
        use windows_sys::Win32::System::JobObjects::{AssignProcessToJobObject, TerminateJobObject};

        const MIB: u64 = 1 << 20;
        let job = windows_job::create_job(768 * MIB).expect("the test job is made");
        let port = windows_job::new_port().expect("the test port is made");
        assert!(windows_job::arm_job(job, port, 7, 192 * MIB));

        let mut child = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Start-Sleep -Milliseconds 500; $held = New-Object byte[] (384MB); Start-Sleep 30",
            ])
            .creation_flags(0x0800_0000)
            .spawn()
            .expect("powershell starts");
        unsafe {
            assert_ne!(AssignProcessToJobObject(job as HANDLE, child.as_raw_handle() as HANDLE), 0);
        }

        let crossing = windows_job::next_limit_crossing(port, 20_000);
        let report = windows_job::crossing_report(job, MemoryLimit::Soft);
        unsafe {
            TerminateJobObject(job as HANDLE, 1);
        }
        let _ = child.wait();
        unsafe {
            CloseHandle(port as HANDLE);
            CloseHandle(job as HANDLE);
        }

        assert_eq!(crossing, Some((MemoryLimit::Soft, 7)));
        assert!(report.contains(&format!("{} powershell.exe", child.id())), "{}", report);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn the_footprint_of_this_process_and_a_child_it_started_is_readable() {
        let mut child = std::process::Command::new("/bin/sleep").arg("30").spawn().expect("sleep starts");
        let me = std::process::id() as i32;
        let child_pid = child.id() as i32;
        let tree = macos_tree::tree(me);
        let own = macos_tree::footprint(me);
        let childs = macos_tree::footprint(child_pid);
        let _ = child.kill();
        let _ = child.wait();

        assert!(tree.contains(&me) && tree.contains(&child_pid), "{:?}", tree);
        assert!(own > 0, "this process's footprint must be readable");
        assert!(childs > 0, "a child's footprint must be readable");
    }
}
