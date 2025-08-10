// Example: Testing NewDistUniFrac with tree and CSV files
// Place this file in examples/ directory and run with: cargo run --example test_unifrac_files

use anndists::dist::distances::{NewDistUniFrac, Distance};
use std::fs;
use std::path::Path;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧬 Testing NewDistUniFrac with your tree and CSV files");
    println!("=" .repeat(60));
    
    // Get file paths from command line or use defaults
    let args: Vec<String> = env::args().collect();
    let tree_file = args.get(1).map(|s| s.as_str()).unwrap_or("tree.nwk");
    let csv_file = args.get(2).map(|s| s.as_str()).unwrap_or("data.csv");
    
    println!("Looking for:");
    println!("  Tree file: {}", tree_file);
    println!("  CSV file:  {}", csv_file);
    println!();
    
    // Check if files exist
    if !Path::new(tree_file).exists() {
        eprintln!("❌ Tree file '{}' not found!", tree_file);
        eprintln!("\nUsage:");
        eprintln!("  cargo run --example test_unifrac_files [tree_file] [csv_file]");
        eprintln!("  cargo run --example test_unifrac_files tree.nwk data.csv");
        eprintln!("\nOr place files in project root as 'tree.nwk' and 'data.csv'");
        return Ok(());
    }
    
    if !Path::new(csv_file).exists() {
        eprintln!("❌ CSV file '{}' not found!", csv_file);
        return Ok(());
    }
    
    // Read tree file
    println!("📁 Reading tree file...");
    let newick_str = fs::read_to_string(tree_file)?;
    let newick_str = newick_str.trim();
    
    println!("✅ Tree loaded ({} characters)", newick_str.len());
    println!("   Preview: {}", 
             if newick_str.len() > 80 { 
                 format!("{}...", &newick_str[..77]) 
             } else { 
                 newick_str.to_string() 
             });
    println!();
    
    // Read and parse CSV file
    println!("📁 Reading CSV file...");
    let csv_content = fs::read_to_string(csv_file)?;
    let lines: Vec<&str> = csv_content.lines().collect();
    
    if lines.is_empty() {
        eprintln!("❌ CSV file is empty");
        return Ok(());
    }
    
    // Parse header (feature names)
    let header = lines[0];
    let feature_names: Vec<String> = header.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    
    println!("✅ Found {} features:", feature_names.len());
    for (i, name) in feature_names.iter().enumerate() {
        if i < 10 {
            println!("   {}. {}", i+1, name);
        } else if i == 10 {
            println!("   ... and {} more", feature_names.len() - 10);
            break;
        }
    }
    println!();
    
    // Parse sample data
    println!("📊 Parsing sample data...");
    let mut samples = Vec::new();
    let mut sample_names = Vec::new();
    
    for (line_num, line) in lines.iter().skip(1).enumerate() {
        let line = line.trim();
        if line.is_empty() { continue; }
        
        let parts: Vec<&str> = line.split(',').collect();
        if parts.is_empty() { continue; }
        
        // Determine if first column is sample name or data
        let (sample_name, data_parts) = if parts.len() > feature_names.len() {
            // Likely has sample names as first column
            (parts[0].trim().to_string(), &parts[1..])
        } else if parts[0].parse::<f32>().is_ok() {
            // First column is numeric data
            (format!("Sample_{}", line_num + 1), parts.as_slice())
        } else {
            // First column is sample name, but count matches
            (parts[0].trim().to_string(), &parts[1..])
        };
        
        // Parse numeric values
        if data_parts.len() != feature_names.len() {
            eprintln!("⚠️  Warning: Sample '{}' has {} values, expected {}", 
                     sample_name, data_parts.len(), feature_names.len());
            continue;
        }
        
        let values: Result<Vec<f32>, _> = data_parts.iter()
            .map(|s| s.trim().parse::<f32>())
            .collect();
        
        match values {
            Ok(vals) => {
                samples.push(vals);
                sample_names.push(sample_name);
            }
            Err(e) => {
                eprintln!("⚠️  Warning: Could not parse sample '{}': {}", sample_name, e);
            }
        }
    }
    
    println!("✅ Parsed {} valid samples:", samples.len());
    for (i, name) in sample_names.iter().enumerate() {
        if i < 5 {
            let sum: f32 = samples[i].iter().sum();
            let nonzero = samples[i].iter().filter(|&&x| x > 0.0).count();
            println!("   {}. {} (sum={:.2}, {} non-zero features)", i+1, name, sum, nonzero);
        } else if i == 5 {
            println!("   ... and {} more", sample_names.len() - 5);
            break;
        }
    }
    
    if samples.is_empty() {
        eprintln!("❌ No valid samples found!");
        return Ok(());
    }
    
    println!();
    
    // Create NewDistUniFrac
    println!("🧬 Creating NewDistUniFrac...");
    let unifrac = match NewDistUniFrac::new(&newick_str, false, feature_names.clone()) {
        Ok(u) => {
            println!("✅ NewDistUniFrac created successfully!");
            println!("   Features: {}", u.num_features());
            println!("   Weighted: {}", u.weighted);
            u
        }
        Err(e) => {
            eprintln!("❌ Error creating NewDistUniFrac: {}", e);
            eprintln!("\n🔧 Troubleshooting:");
            eprintln!("   1. Check that feature names in CSV match tree leaf names exactly");
            eprintln!("   2. Ensure tree is in valid Newick format");
            eprintln!("   3. Verify all feature names exist as leaves in the tree");
            eprintln!("   4. Check for typos or case mismatches");
            return Ok(());
        }
    };
    
    println!();
    
    // Compute distances
    println!("🔢 Computing UniFrac distances...");
    println!("=" .repeat(60));
    
    let num_samples = samples.len();
    let max_display = 10;  // Limit output for readability
    let mut computed = 0;
    
    // Self-distances (should be zero)
    println!("\n🔍 Self-distance test:");
    let self_dist = unifrac.eval(&samples[0], &samples[0]);
    println!("   {} vs itself: {:.8}", sample_names[0], self_dist);
    if self_dist.abs() < 1e-6 {
        println!("   ✅ Self-distance is ~0 ✓");
    } else {
        println!("   ⚠️  Warning: Self-distance should be ~0");
    }
    
    // Pairwise distances
    println!("\n📊 Pairwise distances:");
    for i in 0..num_samples {
        for j in (i + 1)..num_samples {
            if computed >= max_display {
                println!("   ... ({} more pairs)", 
                         (num_samples * (num_samples - 1)) / 2 - computed);
                break;
            }
            
            let distance = unifrac.eval(&samples[i], &samples[j]);
            println!("   {} ↔ {}: {:.6}", sample_names[i], sample_names[j], distance);
            computed += 1;
            
            // Validate distance
            if distance < -1e-10 {
                println!("     ⚠️  Warning: Negative distance!");
            } else if distance > 1.0001 {
                println!("     ⚠️  Warning: Distance > 1!");
            }
        }
        if computed >= max_display { break; }
    }
    
    // Summary statistics
    println!("\n📈 Computing all pairwise distances for statistics...");
    let mut all_distances = Vec::new();
    
    for i in 0..num_samples {
        for j in (i + 1)..num_samples {
            let d = unifrac.eval(&samples[i], &samples[j]);
            all_distances.push(d);
        }
    }
    
    if !all_distances.is_empty() {
        all_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min_dist = all_distances[0];
        let max_dist = all_distances[all_distances.len() - 1];
        let mean_dist = all_distances.iter().sum::<f32>() / all_distances.len() as f32;
        
        println!("📊 Distance Statistics:");
        println!("   Total pairs: {}", all_distances.len());
        println!("   Min distance: {:.6}", min_dist);
        println!("   Max distance: {:.6}", max_dist);
        println!("   Mean distance: {:.6}", mean_dist);
        
        // Distance distribution
        let zeros = all_distances.iter().filter(|&&d| d.abs() < 1e-6).count();
        let small = all_distances.iter().filter(|&&d| d > 1e-6 && d < 0.1).count();
        let medium = all_distances.iter().filter(|&&d| d >= 0.1 && d < 0.5).count();
        let large = all_distances.iter().filter(|&&d| d >= 0.5).count();
        
        println!("   ~0 (identical): {}", zeros);
        println!("   Small (0-0.1): {}", small);
        println!("   Medium (0.1-0.5): {}", medium);
        println!("   Large (0.5+): {}", large);
    }
    
    println!();
    println!("🎉 Analysis complete!");
    println!("=" .repeat(60));
    println!();
    
    Ok(())
}
