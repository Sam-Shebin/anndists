use anndists::dist::distances::{NewDistUniFrac, Distance};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Same setup
    let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
    let feature_names = vec!["T1".to_string(), "T2".to_string(), "T3".to_string(), "T4".to_string()];
    let sample1 = vec![1.0, 2.0, 0.0, 1.0];
    let sample2 = vec![0.0, 1.0, 3.0, 2.0];
    
    // Create NewDistUniFrac instance
    let dist_unifrac = NewDistUniFrac::new(&newick_str, false, feature_names)?;
    
    println!("=== NewDistUniFrac Output Comparison ===\n");
    
    // Method 1: Direct API (what you saw in your tests)
    println!("🔹 Method 1: Direct NewDistUniFrac API");
    let distance1 = dist_unifrac.eval(&sample1, &sample2);
    println!("   Distance: {:.6}", distance1);
    println!("   Features: {}", dist_unifrac.num_features());
    println!("   Weighted: {}", dist_unifrac.weighted);
    
    println!();
    
    // Method 2: Using Distance trait interface (polymorphic)
    println!("🔹 Method 2: Distance Trait Interface");
    let distance_trait: &dyn Distance<f32> = &dist_unifrac;
    let distance2 = distance_trait.eval(&sample1, &sample2);
    println!("   Distance: {:.6}", distance2);
    println!("   (Same result, different interface)");
    
    println!();
    
    // Method 3: Generic function using Distance trait
    fn calculate_distance<D: Distance<f32>>(dist: &D, a: &[f32], b: &[f32]) -> f32 {
        dist.eval(a, b)
    }
    
    println!("🔹 Method 3: Generic Function with Distance Trait");
    let distance3 = calculate_distance(&dist_unifrac, &sample1, &sample2);
    println!("   Distance: {:.6}", distance3);
    println!("   (Can be used with ANY Distance implementation)");
    
    println!();
    println!("✅ All methods produce identical results: {}", 
             distance1 == distance2 && distance2 == distance3);
    
    Ok(())
}
