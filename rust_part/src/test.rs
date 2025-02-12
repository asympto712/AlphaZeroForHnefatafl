    /* 

use std::ffi::CString;
use std::os::raw::c_char;
use winapi::um::libloaderapi::LoadLibraryA;

fn main() {

    let path = CString::new("Path/to/lib/torch_cuda.dll").unwrap();
    
    unsafe {
        LoadLibraryA(path.as_ptr() as *const c_char);
    }

    println!("cuda: {}", tch::Cuda::is_available());
    println!("cudnn: {}", tch::Cuda::cudnn_is_available());

}

*/

fn main() {
    println!("Hello, world!");
}

