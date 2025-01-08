fn main() {
    println!("cargo:rustc-link-search=native=ExternC_UniFrac");
    println!("cargo:rustc-link-lib=dylib=unifrac");

    // Add an rpath to tell the loader to look in ExternC_UniFrac at runtime
    println!("cargo:rustc-link-arg=-Wl,-rpath,./ExternC_UniFrac");
}
