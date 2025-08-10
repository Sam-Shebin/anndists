// Quick test script to verify data files are readable
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Checking data files...");
    
    // Check tree file
    let tree_file = "data/Mouse_gut_zotu_aligned.tre";
    match fs::read_to_string(tree_file) {
        Ok(content) => {
            println!("✅ Tree file readable: {} characters", content.len());
            let preview = if content.len() > 100 { &content[..100] } else { &content };
            println!("   Preview: {}", preview);
        }
        Err(e) => {
            println!("❌ Cannot read tree file: {}", e);
            return Ok(());
        }
    }
    
    // Check counts file
    let counts_file = "data/Mouse_gut_zotu_counts.txt";
    match fs::read_to_string(counts_file) {
        Ok(content) => {
            println!("✅ Counts file readable: {} characters", content.len());
            let lines: Vec<&str> = content.lines().collect();
            println!("   Lines: {}", lines.len());
            if !lines.is_empty() {
                println!("   First line: {}", lines[0]);
                if lines.len() > 1 {
                    println!("   Second line: {}", lines[1]);
                }
            }
        }
        Err(e) => {
            println!("❌ Cannot read counts file: {}", e);
        }
    }
    
    println!("✅ Data file check complete!");
    Ok(())
}
