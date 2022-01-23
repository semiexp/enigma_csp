use super::config;
use super::csugar_cli;

static mut SHARED_ARRAY: Vec<u8> = vec![];

#[no_mangle]
fn run_solver(input: *const u8, len: i32) -> *const u8 {
    let mut input = unsafe { std::slice::from_raw_parts(input, len as usize) };
    let result = csugar_cli::csugar_cli(&mut input, config::Config::default());
    unsafe {
        SHARED_ARRAY.clear();
        SHARED_ARRAY.extend_from_slice(result.as_bytes());
        SHARED_ARRAY.push(0);
        SHARED_ARRAY.as_ptr()
    }
}
