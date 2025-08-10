// Simple test to check compilation
use std::process::Command;

fn main() {
    let output = Command::new("cargo")
        .args(&["test", "test_new_dist_unifrac_from_files", "--", "--nocapture"])
        .current_dir(".")
        .output()
        .expect("Failed to execute command");

    println!("STDOUT:\n{}", String::from_utf8_lossy(&output.stdout));
    println!("STDERR:\n{}", String::from_utf8_lossy(&output.stderr));
    println!("Exit status: {}", output.status);
}
