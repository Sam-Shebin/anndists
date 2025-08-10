// Example: How to get the same numbers using Distance trait

use anndists::dist::distances::{NewDistUniFrac, Distance};

fn main() {
    // Simple example showing Distance trait usage
    let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
    let feature_names = vec!["T1".to_string(), "T2".to_string(), "T3".to_string(), "T4".to_string()];
    
    // Create NewDistUniFrac
    let dist_unifrac = NewDistUniFrac::new(newick_str, false, feature_names).unwrap();
    
    // Sample data (like your mouse gut samples)
    let samples = vec![
        vec![1.0, 2.0, 0.0, 1.0],  // Sample A (like BuuuuA)
        vec![0.0, 1.0, 3.0, 2.0],  // Sample B (like BuuuuB)
        vec![2.0, 0.0, 1.0, 0.0],  // Sample C (like BuuuuC)
        vec![1.0, 1.0, 1.0, 1.0],  // Sample D (like BuuuuD)
    ];
    let sample_names = vec!["SampleA", "SampleB", "SampleC", "SampleD"];
    
    println!("=== Getting Numbers Using Distance Trait ===\\n");
    
    // METHOD 1: Cast to Distance trait
    let distance_trait: &dyn Distance<f32> = &dist_unifrac;
    
    println!("🔹 Using Distance Trait:");
    println!("   {} vs {}: {:.6}", sample_names[0], sample_names[1], distance_trait.eval(&samples[0], &samples[1]));
    println!("   {} vs {}: {:.6}", sample_names[0], sample_names[2], distance_trait.eval(&samples[0], &samples[2]));
    println!("   {} vs {}: {:.6}", sample_names[0], sample_names[3], distance_trait.eval(&samples[0], &samples[3]));
    println!("   Self-distance: {:.8}", distance_trait.eval(&samples[0], &samples[0]));
    
    println!();
    
    // METHOD 2: Generic function
    fn calculate_all_distances<D: Distance<f32>>(
        dist: &D, 
        samples: &[Vec<f32>], 
        names: &[&str]
    ) {
        println!("🔹 Generic Function:");
        for i in 0..3 {
            println!("   {} vs {}: {:.6}", names[0], names[i+1], dist.eval(&samples[0], &samples[i+1]));
        }
        println!("   Self-distance: {:.8}", dist.eval(&samples[0], &samples[0]));
    }
    
    let names: Vec<&str> = sample_names.iter().map(|s| s.as_str()).collect();
    calculate_all_distances(&dist_unifrac, &samples, &names);
    
    println!();
    println!("✅ Both methods give identical results!");
    
    // Verify they match
    let direct = dist_unifrac.eval(&samples[0], &samples[1]);
    let via_trait = distance_trait.eval(&samples[0], &samples[1]);
    println!("\\n🎯 Verification: Direct={:.6}, Trait={:.6}, Equal={}", 
             direct, via_trait, direct == via_trait);
}
