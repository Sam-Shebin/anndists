use anndists::dist::distances::NewDistUniFrac;

fn main() {
    println!("Testing NewDistUniFrac compilation...");
    
    let newick_str = "((A:0.1,B:0.2):0.05,(C:0.3,D:0.4):0.1):0.0;";
    let weighted = false;
    let feature_names = vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()];
    
    match NewDistUniFrac::new(&newick_str, weighted, feature_names) {
        Ok(dist_unifrac) => {
            println!("✅ NewDistUniFrac created successfully!");
            
            let sample1 = vec![1.0, 2.0, 0.0, 1.0];
            let sample2 = vec![0.0, 1.0, 3.0, 2.0];
            
            let distance = dist_unifrac.eval(&sample1, &sample2);
            println!("✅ Distance calculated: {:.6}", distance);
        }
        Err(e) => {
            println!("❌ Error: {:?}", e);
        }
    }
}
