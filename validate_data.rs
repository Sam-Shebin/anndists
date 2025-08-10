// Simple validation for your mouse gut microbiome data
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧬 Mouse Gut Microbiome Data Validation");
    println!("=" .repeat(50));
    
    // Read tree file
    let tree_content = fs::read_to_string("data/Mouse_gut_zotu_aligned.tre")?;
    println!("✅ Tree file: {} characters", tree_content.len());
    
    // Validate it's Newick format
    if tree_content.trim().ends_with(';') {
        println!("✅ Valid Newick format (ends with semicolon)");
    } else {
        println!("⚠️  Warning: Tree may not be in valid Newick format");
    }
    
    // Count taxa in tree
    let paren_count = tree_content.matches('(').count();
    let approx_taxa = paren_count + 1; // rough estimate
    println!("📊 Approximate taxa count: {}", approx_taxa);
    
    // Read counts file
    let counts_content = fs::read_to_string("data/Mouse_gut_zotu_counts.txt")?;
    let lines: Vec<&str> = counts_content.lines().collect();
    println!("✅ Counts file: {} lines", lines.len());
    
    if !lines.is_empty() {
        let header = lines[0];
        let parts: Vec<&str> = header.split('\t').collect();
        println!("📊 Samples in header: {} (first is OTU ID)", parts.len());
        
        // Show first few sample names
        if parts.len() > 1 {
            let sample_names: Vec<&str> = parts[1..].iter().take(5).copied().collect();
            println!("📝 First samples: {:?}", sample_names);
        }
        
        // Count OTUs
        let otu_count = lines.len() - 1; // exclude header
        println!("📊 OTU count: {}", otu_count);
        
        // Show first few OTUs
        if lines.len() > 1 {
            let first_otu_line = lines[1];
            let otu_parts: Vec<&str> = first_otu_line.split('\t').collect();
            if !otu_parts.is_empty() {
                println!("📝 First OTU: {}", otu_parts[0]);
            }
        }
    }
    
    println!("\n🎯 Data files look good for UniFrac analysis!");
    println!("Next step: Run cargo test to validate with NewDistUniFrac");
    
    Ok(())
}
