@echo off
echo Running NewDistUniFrac tests...
cargo test new_dist_unifrac -- --nocapture > test_results.txt 2>&1
echo Test results saved to test_results.txt
type test_results.txt
