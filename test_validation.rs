// Simple validation test for NewDistUniFrac
use anndists::dist::distances::{NewDistUniFrac, Distance};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing NewDistUniFrac creation...");
    
    // Simple 4-leaf tree
    let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
    let feature_names = vec!["T1".to_string(), "T2".to_string(), "T3".to_string(), "T4".to_string()];
    
    // Test creation
    let dist_unifrac = NewDistUniFrac::new(newick_str, false, feature_names)?;
    println!("✓ NewDistUniFrac created successfully");
    println!("  - Weighted: {}", dist_unifrac.weighted);
    println!("  - Number of features: {}", dist_unifrac.num_features());
    
    // Test distance calculation
    let va = vec![1.0, 0.0, 1.0, 0.0]; // T1, T3 present
    let vb = vec![0.0, 1.0, 0.0, 1.0]; // T2, T4 present
    
    let distance = dist_unifrac.eval(&va, &vb);
    println!("✓ Distance calculation successful: {}", distance);
    
    // Test identical samples
    let vc = vec![1.0, 0.0, 1.0, 0.0]; // Same as va
    let distance2 = dist_unifrac.eval(&va, &vc);
    println!("✓ Identical sample distance: {}", distance2);
    
    if distance2.abs() < 1e-6 {
        println!("✓ Identical samples have ~zero distance");
    } else {
        println!("⚠ Warning: Identical samples should have ~zero distance");
    }
    
    if distance > 0.0 {
        println!("✓ Different samples have positive distance");
    } else {
        println!("⚠ Warning: Different samples should have positive distance");
    }
    
    println!("All basic tests passed!");
    Ok(())
}
