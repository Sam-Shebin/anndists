//! Some standard distances as L1, L2, Cosine, Jaccard, Hamming
//! and a structure to enable the user to implement its own distances.
//! For the heavily used case (f32) we provide simd avx2 and std::simd implementations.

#[cfg(feature = "stdsimd")]
use super::distsimd::*;

#[cfg(feature = "simdeez_f")]
use super::disteez::*;
use std::arch::x86_64::*;
/// The trait describing distance.
/// For example for the L1 distance
///
/// pub struct DistL1;
///
/// implement Distance<f32> for DistL1 {
/// }
///
///
/// The L1 and Cosine distance are implemented for u16, i32, i64, f32, f64
///
///
use std::os::raw::c_ulonglong;

use num_traits::float::*;

/// for DistUniFrac (original implementation only)
use anyhow::{anyhow, Result};
use phylotree::tree::{Tree, NodeId};
use std::collections::HashMap;
use log::debug;

// for BitVec used in NewDistUniFrac
use bitvec::prelude::*;

// NewDistUniFrac uses succparen data structures for efficient tree representation

// for DistCFnPtr_UniFrac  
use std::os::raw::{c_char, c_double, c_uint};
use std::slice;

#[allow(unused)]
enum DistKind {
    DistL1(String),
    DistL2(String),
    /// This is the same as Cosine dist but all data L2-normalized to 1.
    DistDot(String),
    DistCosine(String),
    DistHamming(String),
    DistJaccard(String),
    DistHellinger(String),
    DistJeffreys(String),
    DistJensenShannon(String),
    /// UniFrac distance 
    DistUniFrac(String),
    /// To store a distance defined by a C pointer function
    DistCFnPtr,
    /// To store a distance defined by a UniFrac C pointer function, see here: https://github.com/sfiligoi/unifrac-binaries/tree/simple1_250107
    DistUniFracCFFI(String),
    /// Distance defined by a closure
    DistFn,
    /// Distance defined by a fn Rust pointer
    DistPtr,
    DistLevenshtein(String),
    /// used only with reloading only graph data from a previous dump
    DistNoDist(String),
}

/// This is the basic Trait describing a distance. The structure Hnsw can be instantiated by anything
/// satisfying this Trait. The crate provides implmentations for L1, L2 , Cosine, Jaccard, Hamming.
/// For other distances implement the trait possibly with the newtype pattern
pub trait Distance<T: Send + Sync> {
    fn eval(&self, va: &[T], vb: &[T]) -> f32;
}

/// Special forbidden computation distance. It is associated to a unit NoData structure
/// This is a special structure used when we want to only reload the graph from a previous computation
/// possibly from an foreign language (and we do not have access to the original type of data from the foreign language).
#[derive(Default, Copy, Clone)]
pub struct NoDist;

impl<T: Send + Sync> Distance<T> for NoDist {
    fn eval(&self, _va: &[T], _vb: &[T]) -> f32 {
        log::error!("panic error : cannot call eval on NoDist");
        panic!("cannot call distance with NoDist");
    }
} // end impl block for NoDist

/// L1 distance : implemented for i32, f64, i64, u32 , u16 , u8 and with Simd avx2 for f32
#[derive(Default, Copy, Clone)]
pub struct DistL1;

macro_rules! implementL1Distance (
    ($ty:ty) => (

    impl Distance<$ty> for DistL1  {
        fn eval(&self, va:&[$ty], vb: &[$ty]) -> f32 {
            assert_eq!(va.len(), vb.len());
        // RUSTFLAGS = "-C opt-level=3 -C target-cpu=native"
            va.iter().zip(vb.iter()).map(|t| (*t.0 as f32- *t.1 as f32).abs()).sum()
        } // end of compute
    } // end of impl block
    )  // end of pattern matching
);

implementL1Distance!(i32);
implementL1Distance!(f64);
implementL1Distance!(i64);
implementL1Distance!(u32);
implementL1Distance!(u16);
implementL1Distance!(u8);

impl Distance<f32> for DistL1 {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        //
        cfg_if::cfg_if! {
        if #[cfg(feature = "simdeez_f")] {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                if is_x86_feature_detected!("avx2") {
                    return unsafe {distance_l1_f32_avx2(va,vb)};
                }
                else {
                    assert_eq!(va.len(), vb.len());
                    va.iter().zip(vb.iter()).map(|t| (*t.0 as f32- *t.1 as f32).abs()).sum()
                }
            }
        }
        else if #[cfg(feature = "stdsimd")] {
            distance_l1_f32_simd(va,vb)
        }
        else {
            va.iter().zip(vb.iter()).map(|t| (*t.0 as f32- *t.1 as f32).abs()).sum()
        }
        } // end cfg_if
    } // end of eval
} // end impl Distance<f32> for DistL1

//========================================================================

/// L2 distance : implemented for i32, f64, i64, u32 , u16 , u8 and with Simd avx2 for f32
#[derive(Default, Copy, Clone)]
pub struct DistL2;

macro_rules! implementL2Distance (
    ($ty:ty) => (

    impl Distance<$ty> for DistL2  {
        fn eval(&self, va:&[$ty], vb: &[$ty]) -> f32 {
            assert_eq!(va.len(), vb.len());
            let norm : f32 = va.iter().zip(vb.iter()).map(|t| (*t.0 as f32- *t.1 as f32) * (*t.0 as f32- *t.1 as f32)).sum();
            norm.sqrt()
        } // end of compute
    } // end of impl block
    )  // end of pattern matching
);

//implementL2Distance!(f32);
implementL2Distance!(i32);
implementL2Distance!(f64);
implementL2Distance!(i64);
implementL2Distance!(u32);
implementL2Distance!(u16);
implementL2Distance!(u8);

#[allow(unused)]
// base scalar l2 for f32
fn scalar_l2_f32(va: &[f32], vb: &[f32]) -> f32 {
    let norm: f32 = va
        .iter()
        .zip(vb.iter())
        .map(|t| (*t.0 as f32 - *t.1 as f32) * (*t.0 as f32 - *t.1 as f32))
        .sum();
    assert!(norm >= 0.);
    return norm.sqrt();
}

impl Distance<f32> for DistL2 {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        //
        cfg_if::cfg_if! {
            if #[cfg(feature = "simdeez_f")] {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    return unsafe { distance_l2_f32_avx2(va, vb) };
                }
                else {
                    return scalar_l2_f32(&va, &vb);
                }
            }
            } else if #[cfg(feature = "stdsimd")] {
                return distance_l2_f32_simd(va, vb);
            }
            else {
                let norm = scalar_l2_f32(&va, &vb);
                return norm;
            }
        }
    } // end of eval
} // end impl Distance<f32> for DistL2

//=========================================================================

/// Cosine distance : implemented for f32, f64, i64, i32 , u16
#[derive(Default, Copy, Clone)]
pub struct DistCosine;

macro_rules! implementCosDistance(
    ($ty:ty) => (
     impl Distance<$ty> for DistCosine  {
        fn eval(&self, va:&[$ty], vb: &[$ty]) -> f32 {
            assert_eq!(va.len(), vb.len());
            //
            let dist:f32;
            let zero:f64 = 0.;
            // to // by rayon
            let res = va.iter().zip(vb.iter()).map(|t| ((*t.0 * *t.1) as f64, (*t.0 * *t.0) as f64, (*t.1 * *t.1) as f64)).
                fold((0., 0., 0.), |acc , t| (acc.0 + t.0, acc.1 + t.1, acc.2 + t.2));
            //
            if res.1 > zero && res.2 > zero {
                let dist_unchecked = 1. - res.0 / (res.1 * res.2).sqrt();
                assert!(dist_unchecked >= - 0.00002);
                dist = dist_unchecked.max(0.) as f32;
            }
            else {
                dist = 0.;
            }
            //
            return dist;
        } // end of function
     } // end of impl block
    ) // end of matching
);

implementCosDistance!(f32);
implementCosDistance!(f64);
implementCosDistance!(i64);
implementCosDistance!(i32);
implementCosDistance!(u16);

//=========================================================================

/// This is essentially the Cosine distance but we suppose
/// all vectors (graph construction and request vectors have been l2 normalized to unity
/// BEFORE INSERTING in  HNSW!.   
/// No control is made, so it is the user responsability to send normalized vectors
/// everywhere in inserting and searching.
///
/// In large dimensions (hundreds) this pre-normalization spare cpu time.  
/// At low dimensions (a few ten's there is not a significant gain).  
/// This distance makes sense only for f16, f32 or f64
/// We provide for avx2 implementations for f32 that provides consequent gains
/// in large dimensions

#[derive(Default, Copy, Clone)]
pub struct DistDot;

#[allow(unused)]
macro_rules! implementDotDistance(
    ($ty:ty) => (
     impl Distance<$ty> for DistDot  {
        fn eval(&self, va:&[$ty], vb: &[$ty]) -> f32 {
            assert_eq!(va.len(), vb.len());
            //
            let zero:f32 = 0f32;
            // to // by rayon
            let dot = va.iter().zip(vb.iter()).map(|t| (*t.0 * *t.1) as f32).fold(0., |acc , t| (acc + t));
            //
            assert(dot <= 1.);
            return  1. - dot;
        } // end of function
      } // end of impl block
    ) // end of matching
);

#[allow(unused)]
fn scalar_dot_f32(va: &[f32], vb: &[f32]) -> f32 {
    let dot = 1.
        - va.iter()
            .zip(vb.iter())
            .map(|t| (*t.0 * *t.1) as f32)
            .fold(0., |acc, t| (acc + t));
    assert!(dot >= 0.);
    dot
}

impl Distance<f32> for DistDot {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        //
        cfg_if::cfg_if! {
            if #[cfg(feature = "simdeez_f")] {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        return unsafe { distance_dot_f32_avx2(va, vb) };
                    } else if is_x86_feature_detected!("sse2") {
                        return unsafe { distance_dot_f32_sse2(va, vb) };
                    }
                    else {
                        return scalar_dot_f32(va, vb);
                    }
                } // end x86
            } else if #[cfg(feature = "stdsimd")] {
                return distance_dot_f32_simd_iter(va,vb);
            }
            else {
                return scalar_dot_f32(va, vb);
            }
        }
    } // end of eval
}

pub fn l2_normalize(va: &mut [f32]) {
    let l2norm = va.iter().map(|t| (*t * *t) as f32).sum::<f32>().sqrt();
    if l2norm > 0. {
        for i in 0..va.len() {
            va[i] = va[i] / l2norm;
        }
    }
}

//=======================================================================================

///
/// A structure to compute Hellinger distance between probalilities.
/// Vector must be >= 0 and normalized to 1.
///   
/// The distance computation does not check that
/// and in fact simplifies the expression of distance assuming vectors are positive and L1 normalised to 1.
/// The user must enforce these conditions before  inserting otherwise results will be meaningless
/// at best or code will panic!
///
/// For f32 a simd implementation is provided if avx2 is detected.
#[derive(Default, Copy, Clone)]
pub struct DistHellinger;

// default implementation
macro_rules! implementHellingerDistance (
    ($ty:ty) => (

    impl Distance<$ty> for DistHellinger {
        fn eval(&self, va:&[$ty], vb: &[$ty]) -> f32 {
            assert_eq!(va.len(), vb.len());
        // RUSTFLAGS = "-C opt-level=3 -C target-cpu=native"
        // to // by rayon
            let mut dist = va.iter().zip(vb.iter()).map(|t| ((*t.0).sqrt() * (*t.1).sqrt()) as f32).fold(0., |acc , t| (acc + t*t));
            dist = (1. - dist).sqrt();
            dist
        } // end of compute
    } // end of impl block
    )  // end of pattern matching
);

implementHellingerDistance!(f64);

impl Distance<f32> for DistHellinger {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        //
        #[cfg(feature = "simdeez_f")]
        {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    return unsafe { distance_hellinger_f32_avx2(va, vb) };
                }
            }
        }
        let mut dist = va
            .iter()
            .zip(vb.iter())
            .map(|t| ((*t.0).sqrt() * (*t.1).sqrt()) as f32)
            .fold(0., |acc, t| (acc + t));
        // if too far away from >= panic else reset!
        assert!(1. - dist >= -0.000001);
        dist = (1. - dist).max(0.).sqrt();
        dist
    } // end of eval
}

//=======================================================================================

///
/// A structure to compute Jeffreys divergence between probalilities.
/// If p and q are 2 probability distributions
/// the "distance" is computed as:
///   sum (p\[i\] - q\[i\]) * ln(p\[i\]/q\[i\])
///
/// To take care of null probabilities in the formula we use  max(x\[i\],1.E-30)
/// for x = p and q in the log compuations
///   
/// Vector must be >= 0 and normalized to 1!  
/// The distance computation does not check that.
/// The user must enforce these conditions before inserting in the hnws structure,
/// otherwise results will be meaningless at best or code will panic!
///
/// For f32 a simd implementation is provided if avx2 is detected.
#[derive(Default, Copy, Clone)]
pub struct DistJeffreys;

pub const M_MIN: f32 = 1.0e-30;

// default implementation
macro_rules! implementJeffreysDistance (
    ($ty:ty) => (

    impl Distance<$ty> for DistJeffreys {
        fn eval(&self, va:&[$ty], vb: &[$ty]) -> f32 {
        // RUSTFLAGS = "-C opt-level=3 -C target-cpu=native"
        let dist = va.iter().zip(vb.iter()).map(|t| (*t.0 - *t.1) * ((*t.0).max(M_MIN as f64)/ (*t.1).max(M_MIN as f64)).ln() as f64).fold(0., |acc , t| (acc + t*t));
        dist as f32
        } // end of compute
    } // end of impl block
    )  // end of pattern matching
);

implementJeffreysDistance!(f64);

impl Distance<f32> for DistJeffreys {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        //
        #[cfg(feature = "simdeez_f")]
        {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    return unsafe { distance_jeffreys_f32_avx2(va, vb) };
                }
            }
        }
        let dist = va
            .iter()
            .zip(vb.iter())
            .map(|t| (*t.0 - *t.1) * ((*t.0).max(M_MIN) / (*t.1).max(M_MIN)).ln() as f32)
            .fold(0., |acc, t| (acc + t));
        dist
    } // end of eval
}

//=======================================================================================

/// Jensen-Shannon distance.  
/// It is defined as the **square root** of the  Jensen–Shannon divergence and is a metric.
/// Vector must be >= 0 and normalized to 1!
/// **The distance computation does not check that**.
#[derive(Default, Copy, Clone)]
pub struct DistJensenShannon;

macro_rules! implementDistJensenShannon (

    ($ty:ty) => (
        impl Distance<$ty> for DistJensenShannon {
            fn eval(&self, va:&[$ty], vb: &[$ty]) -> f32 {
                let mut dist = 0.;
                //
                assert_eq!(va.len(), vb.len());
                //
                for i in 0..va.len() {
                    let mean_ab = 0.5 * (va[i] + vb[i]);
                    if va[i] > 0. {
                        dist += va[i] * (va[i]/mean_ab).ln();
                    }
                    if vb[i] > 0. {
                        dist += vb[i] * (vb[i]/mean_ab).ln();
                    }
                }
                (0.5 * dist).sqrt() as f32
            } // end eval
        }  // end impl Distance<$ty>
    )  // end of pattern matching on ty
);

implementDistJensenShannon!(f64);
implementDistJensenShannon!(f32);

//=======================================================================================

/// Hamming distance. Implemented for u8, u16, u32, i32 and i16
/// The distance returned is normalized by length of slices, so it is between 0. and 1.  
///
/// A special implementation for f64 is made but exclusively dedicated to SuperMinHash usage in crate [probminhash](https://crates.io/crates/probminhash).  
/// It could be made generic with the PartialEq implementation for f64 and f32 in unsable source of Rust
#[derive(Default, Copy, Clone)]
pub struct DistHamming;

macro_rules! implementHammingDistance (
    ($ty:ty) => (

    impl Distance<$ty> for DistHamming  {
        fn eval(&self, va:&[$ty], vb: &[$ty]) -> f32 {
        // RUSTFLAGS = "-C opt-level=3 -C target-cpu=native"
            assert_eq!(va.len(), vb.len());
            let norm : f32 = va.iter().zip(vb.iter()).filter(|t| t.0 != t.1).count() as f32;
            norm / va.len() as f32
        } // end of compute
    } // end of impl block
    )  // end of pattern matching
);

impl Distance<i32> for DistHamming {
    fn eval(&self, va: &[i32], vb: &[i32]) -> f32 {
        //
        #[cfg(feature = "simdeez_f")]
        {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    return unsafe { distance_hamming_i32_avx2(va, vb) };
                }
            }
        }
        assert_eq!(va.len(), vb.len());
        let dist: f32 = va.iter().zip(vb.iter()).filter(|t| t.0 != t.1).count() as f32;
        dist / va.len() as f32
    } // end of eval
} // end implementation Distance<i32>

/// This implementation is dedicated to SuperMinHash algorithm in crate [probminhash](https://crates.io/crates/probminhash).  
/// Could be made generic with unstable source as there is implementation of PartialEq for f64
impl Distance<f64> for DistHamming {
    fn eval(&self, va: &[f64], vb: &[f64]) -> f32 {
        /*   Tests show that it is slower than basic method!!!
        #[cfg(feature = "simdeez_f")] {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                if is_x86_feature_detected!("avx2") {
                    log::trace!("calling distance_hamming_f64_avx2");
                    return unsafe { distance_hamming_f64_avx2(va,vb) };
                }
            }
        }
        */
        //
        assert_eq!(va.len(), vb.len());
        let dist: usize = va.iter().zip(vb.iter()).filter(|t| t.0 != t.1).count();
        (dist as f64 / va.len() as f64) as f32
    } // end of eval
} // end implementation Distance<f64>

//

/// This implementation is dedicated to SuperMinHash algorithm in crate [probminhash](https://crates.io/crates/probminhash).  
/// Could be made generic with unstable source as there is implementation of PartialEq for f32
impl Distance<f32> for DistHamming {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        cfg_if::cfg_if! {
            if #[cfg(feature = "stdsimd")] {
                return distance_jaccard_f32_16_simd(va,vb);
            }
            else {
                assert_eq!(va.len(), vb.len());
                let dist : usize = va.iter().zip(vb.iter()).filter(|t| t.0 != t.1).count();
                (dist as f64 / va.len() as f64) as f32
            }
        }
    } // end of eval
} // end implementation Distance<f32>

//

#[cfg(feature = "stdsimd")]
impl Distance<u32> for DistHamming {
    fn eval(&self, va: &[u32], vb: &[u32]) -> f32 {
        //
        return distance_jaccard_u32_16_simd(va, vb);
    } // end of eval
} // end implementation Distance<u32>

//

#[cfg(feature = "stdsimd")]
impl Distance<u64> for DistHamming {
    fn eval(&self, va: &[u64], vb: &[u64]) -> f32 {
        return distance_jaccard_u64_8_simd(va, vb);
    } // end of eval
} // end implementation Distance<u64>

// i32 is implmeented by simd
implementHammingDistance!(u8);
implementHammingDistance!(u16);

#[cfg(not(feature = "stdsimd"))]
implementHammingDistance!(u32);

#[cfg(not(feature = "stdsimd"))]
implementHammingDistance!(u64);

implementHammingDistance!(i16);

//====================================================================================
//   Jaccard Distance

/// Jaccard distance. Implemented for u8, u16 , u32.
#[derive(Default, Copy, Clone)]
pub struct DistJaccard;

// contruct a 2-uple accumulator that has sum of max in first component , and sum of min in 2 component
// stay in integer as long as possible
// Note : summing u32 coming from hash values can overflow! We must go up to u64 for additions!
macro_rules! implementJaccardDistance (
    ($ty:ty) => (

    impl Distance<$ty> for DistJaccard  {
        fn eval(&self, va:&[$ty], vb: &[$ty]) -> f32 {
            let (max,min) : (u64, u64) = va.iter().zip(vb.iter()).fold((0u64,0u64), |acc, t| if t.0 > t.1 {
                                (acc.0 + *t.0 as u64, acc.1 + *t.1 as u64) }
                        else {
                                (acc.0 + *t.1 as u64 , acc.1 + *t.0 as u64)
                             }
            );
            if max > 0 {
                let dist = 1. - (min  as f64)/ (max as f64);
                assert!(dist >= 0.);
                dist as f32
            }
            else {
                0.
            }
        } // end of compute
    } // end of impl block
    )  // end of pattern matching
);

implementJaccardDistance!(u8);
implementJaccardDistance!(u16);
implementJaccardDistance!(u32);

// ==========================================================================================

/// Levenshtein distance. Implemented for u16
#[derive(Default, Copy, Clone)]
pub struct DistLevenshtein;
impl Distance<u16> for DistLevenshtein {
    fn eval(&self, a: &[u16], b: &[u16]) -> f32 {
        let len_a = a.len();
        let len_b = b.len();
        if len_a < len_b {
            return self.eval(b, a);
        }
        // handle special case of 0 length
        if len_a == 0 {
            return len_b as f32;
        } else if len_b == 0 {
            return len_a as f32;
        }

        let len_b = len_b + 1;

        let mut pre;
        let mut tmp;
        let mut cur = vec![0; len_b];

        // initialize string b
        for i in 1..len_b {
            cur[i] = i;
        }

        // calculate edit distance
        for (i, ca) in a.iter().enumerate() {
            // get first column for this row
            pre = cur[0];
            cur[0] = i + 1;
            for (j, cb) in b.iter().enumerate() {
                tmp = cur[j + 1];
                cur[j + 1] = std::cmp::min(
                    // deletion
                    tmp + 1,
                    std::cmp::min(
                        // insertion
                        cur[j] + 1,
                        // match or substitution
                        pre + if ca == cb { 0 } else { 1 },
                    ),
                );
                pre = tmp;
            }
        }
        let res = cur[len_b - 1] as f32;
        return res;
    }
}


/// DistUniFrac
#[derive(Default, Clone)]
pub struct DistUniFrac {
    /// Weighted or unweighted
    weighted: bool,
    /// tint[i] = parent of node i
    tint: Vec<usize>,
    /// lint[i] = length of edge from node i up to tint[i]
    lint: Vec<f32>,
    /// Postorder nodes
    nodes_in_order: Vec<usize>,
    /// node_name_map[node_in_order_idx] = Some("T4") or whatever
    node_name_map: Vec<Option<String>>,
    /// leaf_map: "T4" -> postorder index
    leaf_map: HashMap<String, usize>,
    /// total tree length (not used to normalize in this example)
    total_tree_length: f32,
    /// Feature names in the same order as va,vb
    feature_names: Vec<String>,
}

impl DistUniFrac {
    /// Build DistUniFrac from the given `newick_str` (no re-root),
    /// a boolean `weighted` for Weighted or Unweighted, plus your feature names.
    ///
    /// *Important*: We do not compress or re-root the tree. This ensures T4's path
    pub fn new(
        newick_str: &str,
        weighted: bool,
        feature_names: Vec<String>,
    ) -> Result<Self> {
        // Parse the tree from the Newick string
        let tree = Tree::from_newick(newick_str)?;

        // Build arrays (same approach as your old code)
        let (tint, lint, nodes_in_order, node_name_map) = build_tint_lint(&tree)?;

        let leaf_map = build_leaf_map(&tree, &node_name_map)?;

        let total_tree_length: f32 = lint.iter().sum();

        Ok(Self {
            weighted,
            tint,
            lint,
            nodes_in_order,
            node_name_map,
            leaf_map,
            total_tree_length,
            feature_names,
        })
    }
}

/// Implement the Distance<f32> trait
impl Distance<f32> for DistUniFrac {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        debug!(
            "DistUniFrac eval called: weighted={}, #features={}",
            self.weighted,
            self.feature_names.len()
        );
        if self.weighted {
            compute_unifrac_for_pair_weighted_bitwise(
                &self.tint,
                &self.lint,
                &self.nodes_in_order,
                &self.leaf_map,
                &self.feature_names,
                va,
                vb,
            )
        } else {
            compute_unifrac_for_pair_unweighted_bitwise(
                &self.tint,
                &self.lint,
                &self.nodes_in_order,
                &self.leaf_map,
                &self.feature_names,
                va,
                vb,
            )
        }
    }
}

//--------------------------------------------------------------------------------------//
//--------------------------------------------------------------------------------------//
fn build_tint_lint(tree: &Tree) -> Result<(Vec<usize>, Vec<f32>, Vec<usize>, Vec<Option<String>>)> {
    let root = tree.get_root().map_err(|_| anyhow!("Tree has no root"))?;
    let postord = tree.postorder(&root)?; 
    debug!("postord = {:?}", postord);
    let num_nodes = postord.len();

    // node_id -> postorder index
    let mut pos_map = vec![0; tree.size()];
    for (i, &nid) in postord.iter().enumerate() {
        pos_map[nid] = i;
    }

    let mut tint = vec![0; num_nodes];
    let mut lint = vec![0.0; num_nodes];
    let mut node_name_map = vec![None; num_nodes];

    // Mark the root as its own ancestor
    let root_idx = pos_map[root];
    tint[root_idx] = root_idx;
    lint[root_idx] = 0.0;

    // Fill in parent/edge arrays
    for &nid in &postord {
        let node = tree.get(&nid)?;
        if let Some(name) = &node.name {
            node_name_map[pos_map[nid]] = Some(name.clone());
        }
        if nid != root {
            let p = node.parent.ok_or_else(|| anyhow!("Node has no parent but is not root"))?;
            tint[pos_map[nid]] = pos_map[p];
            lint[pos_map[nid]] = node.parent_edge.unwrap_or(0.0) as f32;
        }
    }
    Ok((tint, lint, postord, node_name_map))
}

fn build_leaf_map(
    tree: &Tree,
    node_name_map: &[Option<String>],
) -> Result<HashMap<String, usize>> {
    let root = tree.get_root().map_err(|_| anyhow!("Tree has no root"))?;
    let postord = tree.postorder(&root)?;
    debug!("postord = {:?}", postord);
    let mut pos_map = vec![0; tree.size()];
    for (i, &nid) in postord.iter().enumerate() {
        pos_map[nid] = i;
    }

    let mut leaf_map = HashMap::new();
    for l in tree.get_leaves() {
        let node = tree.get(&l)?;
        if node.is_tip() {
            if let Some(name) = &node.name {
                let idx = pos_map[l];
                debug!("   => recognized tip='{}', postord_idx={}", name, idx);
                leaf_map.insert(name.clone(), idx);
            }
        }
    }
    Ok(leaf_map)
}

// start of NewDistUniFrac


/// Parse Newick string directly to succparen data structures without using phylotree
fn parse_newick_to_succparen(
    newick_str: &str,
    feature_names: &[String],
) -> Result<(Vec<usize>, Vec<Vec<usize>>, Vec<f32>, Vec<usize>)> {
    let mut nodes = Vec::new();
    let mut stack = Vec::new();
    let mut current_node_id = 0;
    let mut chars = newick_str.trim_end_matches(';').chars().peekable();

    #[derive(Debug, Clone)]
    struct ParseNode {
        id: usize,
        name: Option<String>,
        branch_length: f32,
        children: Vec<usize>,
        parent: Option<usize>,
    }

    while let Some(ch) = chars.next() {
        match ch {
            '(' => {
                let node = ParseNode {
                    id: current_node_id,
                    name: None,
                    branch_length: 0.0,
                    children: Vec::new(),
                    parent: None,
                };
                stack.push(current_node_id);
                nodes.push(node);
                current_node_id += 1;
            }
            ')' => {
                let node_id = stack.pop().ok_or_else(|| anyhow!("Unmatched closing parenthesis"))?;
                while chars.peek() == Some(&' ') { chars.next(); }
                let mut name = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch == ':' || ch == ',' || ch == ')' || ch == ';' { break; }
                    name.push(chars.next().unwrap());
                }
                if !name.is_empty() {
                    nodes[node_id].name = Some(name.trim().to_string());
                }
                if chars.peek() == Some(&':') {
                    chars.next();
                    let mut length_str = String::new();
                    while let Some(&ch) = chars.peek() {
                        if ch == ',' || ch == ')' || ch == ';' { break; }
                        length_str.push(chars.next().unwrap());
                    }
                    if let Ok(length) = length_str.trim().parse::<f32>() {
                        nodes[node_id].branch_length = length;
                    }
                }
            }
            ',' | ' ' => {}
            _ => {
                let mut name = String::new();
                name.push(ch);
                while let Some(&ch) = chars.peek() {
                    if ch == ':' || ch == ',' || ch == ')' || ch == ';' { break; }
                    name.push(chars.next().unwrap());
                }
                let mut leaf_node = ParseNode {
                    id: current_node_id,
                    name: Some(name.trim().to_string()),
                    branch_length: 0.0,
                    children: Vec::new(),
                    parent: None,
                };
                if chars.peek() == Some(&':') {
                    chars.next();
                    let mut length_str = String::new();
                    while let Some(&ch) = chars.peek() {
                        if ch == ',' || ch == ')' || ch == ';' { break; }
                        length_str.push(chars.next().unwrap());
                    }
                    if let Ok(length) = length_str.trim().parse::<f32>() {
                        leaf_node.branch_length = length;
                    }
                }
                if let Some(&parent_id) = stack.last() {
                    nodes[parent_id].children.push(current_node_id);
                    leaf_node.parent = Some(parent_id);
                }
                nodes.push(leaf_node);
                current_node_id += 1;
            }
        }
    }

    for i in 0..nodes.len() {
        for &child_id in &nodes[i].children.clone() {
            if child_id < nodes.len() {
                nodes[child_id].parent = Some(i);
            }
        }
    }

    let root_id = 0;
    let mut post = Vec::new();
    let mut visited = vec![false; nodes.len()];
    fn postorder_visit(node_id: usize, nodes: &[ParseNode], visited: &mut [bool], post: &mut Vec<usize>) {
        if visited[node_id] { return; }
        visited[node_id] = true;
        for &child_id in &nodes[node_id].children {
            postorder_visit(child_id, nodes, visited, post);
        }
        post.push(node_id);
    }
    postorder_visit(root_id, &nodes, &mut visited, &mut post);

    let mut kids = vec![Vec::new(); nodes.len()];
    for node in &nodes {
        kids[node.id] = node.children.clone();
    }

    let mut lens = vec![0.0f32; nodes.len()];
    for node in &nodes {
        lens[node.id] = node.branch_length;
    }

    let mut leaf_ids = Vec::new();
    for node in &nodes {
        if node.children.is_empty() {
            if let Some(ref name) = node.name {
                if feature_names.iter().any(|f| f == name) {
                    leaf_ids.push(node.id);
                }
            }
        }
    }

    // No longer need phylotree Tree - we use our own succinct data structures
    Ok((post, kids, lens, leaf_ids))
}

#[derive(Default, Clone)]
pub struct NewDistUniFrac {
    pub weighted: bool,
    pub post: Vec<usize>,
    pub kids: Vec<Vec<usize>>,
    pub lens: Vec<f32>,
    pub leaf_ids: Vec<usize>,
    pub feature_names: Vec<String>,
}

impl Distance<f32> for NewDistUniFrac {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        let a: BitVec<u8, Lsb0> = va.iter().map(|&x| x > 0.0).collect();
        let b: BitVec<u8, Lsb0> = vb.iter().map(|&x| x > 0.0).collect();
        unifrac_pair(&self.post, &self.kids, &self.lens, &self.leaf_ids, &a, &b) as f32
    }
}

impl NewDistUniFrac {
    pub fn new(newick_str: &str, weighted: bool, feature_names: Vec<String>) -> Result<Self> {
        let (post, kids, lens, leaf_ids) = parse_newick_to_succparen(newick_str, &feature_names)?;
        Ok(Self { weighted, post, kids, lens, leaf_ids, feature_names })
    }

    pub fn from_files(tree_file: &str, weighted: bool, feature_names: Vec<String>) -> Result<Self> {
        let newick_str = std::fs::read_to_string(tree_file)
            .map_err(|e| anyhow!("Failed to read tree file '{}': {}", tree_file, e))?;
        Self::new(&newick_str, weighted, feature_names)
    }

    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    pub fn num_features(&self) -> usize {
        self.feature_names.len()
    }
}

fn unifrac_pair(
    post: &[usize],
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    a: &BitVec<u8, Lsb0>,
    b: &BitVec<u8, Lsb0>,
) -> f64 {
    const A_BIT: u8 = 0b01;
    const B_BIT: u8 = 0b10;
    let mut mask = vec![0u8; lens.len()];
    for (leaf_pos, &nid) in leaf_ids.iter().enumerate() {
        if a[leaf_pos] { mask[nid] |= A_BIT; }
        if b[leaf_pos] { mask[nid] |= B_BIT; }
    }
    for &v in post {
        for &c in &kids[v] {
            mask[v] |= mask[c];
        }
    }
    let (mut shared, mut union) = (0.0, 0.0);
    for &v in post {
        let m = mask[v];
        if m == 0 { continue }
        let len = lens[v] as f64;
        if m == A_BIT || m == B_BIT { union += len; }
        else { shared += len; union += len; }
    }
    if union == 0.0 { 0.0 } else { 1.0 - shared / union }
}

// End of NewDistUniFrac

/// Build a bitmask for data[i] > 0.0 using AVX2 intrinsics. 
/// # Safety
/// - This function is marked `unsafe` because it uses `target_feature(enable = "avx2")` 
///   and raw pointer arithmetic.
/// - The caller must ensure the CPU supports AVX2.
/// 
/// # Parameters
/// - `data`: slice of f32 values (length can be very large).
/// # Returns
/// - A `Vec<u64>` with `(data.len() + 63) / 64` elements.
///   Each bit in the `u64` corresponds to one element in `data`,
///   set if `data[i] > 0.0`.
#[target_feature(enable = "avx2")]
pub unsafe fn make_presence_mask_f32_avx2(data: &[f32]) -> Vec<u64> {
    let len = data.len();
    let num_chunks = (len + 63) / 64;
    let mut mask = vec![0u64; num_chunks];
    // We'll process 8 floats at a time (AVX2 = 256 bits).
    const STRIDE: usize = 8;
    let blocks = len / STRIDE;
    let remainder = len % STRIDE;
    let ptr = data.as_ptr();
    // For each block of 8 floats
    for blk_idx in 0..blocks {
        let offset = blk_idx * STRIDE;
        // Load 8 consecutive f32 values
        // _mm256_loadu_ps => unaligned load is usually fine on modern x86
        let v = _mm256_loadu_ps(ptr.add(offset));
        // Compare each float in `v` to 0.0 => result bits set to 1 if > 0.0
        let gt_mask = _mm256_cmp_ps(v, _mm256_set1_ps(0.0), _CMP_GT_OQ);
        // `_mm256_movemask_ps` extracts the top bit of each float comparison 
        // into an 8-bit integer: 1 bit per float, 0..7
        let bitmask = _mm256_movemask_ps(gt_mask) as u32; // 8 bits used

        // Now we need to place each bit into the appropriate position in `mask`.
        // The i-th bit in `bitmask` corresponds to data[offset + i].
        // So for i in 0..8, if that bit is set, set the corresponding bit in `mask`.
        if bitmask == 0 {
            // No bits set in this block => skip
            continue;
        }
        for i in 0..STRIDE {
            let global_idx = offset + i;
            // Check if bit i is set
            if (bitmask & (1 << i)) != 0 {
                let chunk_idx = global_idx / 64;
                let bit_idx   = global_idx % 64;
                mask[chunk_idx] |= 1 << bit_idx;
            }
        }
    }
    // Handle any leftover floats at the tail
    let tail_start = blocks * STRIDE;
    if remainder > 0 {
        for i in tail_start..(tail_start + remainder) {
            if *data.get_unchecked(i) > 0.0 {
                let chunk_idx = i / 64;
                let bit_idx   = i % 64;
                mask[chunk_idx] |= 1 << bit_idx;
            }
        }
    }
    mask
}

/// Fallback scalar version, for reference.
fn make_presence_mask_f32_scalar(data: &[f32]) -> Vec<u64> {
    let num_chunks = (data.len() + 63) / 64;
    let mut mask = vec![0u64; num_chunks];
    for (i, &val) in data.iter().enumerate() {
        if val > 0.0 {
            let chunk_idx = i / 64;
            let bit_idx = i % 64;
            mask[chunk_idx] |= 1 << bit_idx;
        }
    }
    mask
}

fn make_presence_mask_f32(data: &[f32]) -> Vec<u64> {
    if is_x86_feature_detected!("avx2") {
        unsafe { make_presence_mask_f32_avx2(data) }
    } else {
        make_presence_mask_f32_scalar(data)
    }
}

fn or_masks(a: &[u64], b: &[u64]) -> Vec<u64> {
    a.iter().zip(b.iter()).map(|(x, y)| x | y).collect()
}

fn extract_set_bits(m: &[u64]) -> Vec<usize> {
    let mut indices = Vec::new();
    for (chunk_idx, &chunk) in m.iter().enumerate() {
        if chunk == 0 {
            continue;
        }
        let base = chunk_idx * 64;
        let mut c = chunk;
        while c != 0 {
            let bit_index = c.trailing_zeros() as usize;
            indices.push(base + bit_index);
            c &= !(1 << bit_index);
        }
    }
    indices
}

//--------------------------------------------------------------------------------------//
//  Bitwise skip-zero logic for unweighted
//--------------------------------------------------------------------------------------//

fn compute_unifrac_for_pair_unweighted_bitwise(
    tint: &[usize],
    lint: &[f32],
    nodes_in_order: &[usize],
    leaf_map: &HashMap<String, usize>,
    feature_names: &[String],
    va: &[f32],
    vb: &[f32],
) -> f32 {
    debug!("=== compute_unifrac_for_pair_unweighted_bitwise ===");
    let num_nodes = nodes_in_order.len();
    let mut partial_sums = vec![0.0; num_nodes];

    // 1) Build presence masks for va, vb
    let mask_a = make_presence_mask_f32(va);
    let mask_b = make_presence_mask_f32(vb);

    // 2) Combine => find non-zero indices
    let combined = or_masks(&mask_a, &mask_b);
    let non_zero_indices = extract_set_bits(&combined);
    debug!("non_zero_indices = {:?}", non_zero_indices);

    // 3) Create local arrays
    let mut local_a = Vec::with_capacity(non_zero_indices.len());
    let mut local_b = Vec::with_capacity(non_zero_indices.len());
    let mut local_feats = Vec::with_capacity(non_zero_indices.len());

    for &idx in &non_zero_indices {
        local_a.push(va[idx]);
        local_b.push(vb[idx]);
        local_feats.push(&feature_names[idx]);
    }

    // 4) Convert to presence/absence, then sum
    let mut sum_a = 0.0;
    let mut sum_b = 0.0;
    for i in 0..local_a.len() {
        if local_a[i] > 0.0 {
            local_a[i] = 1.0;
            sum_a += 1.0;
        } else {
            local_a[i] = 0.0;
        }
        if local_b[i] > 0.0 {
            local_b[i] = 1.0;
            sum_b += 1.0;
        } else {
            local_b[i] = 0.0;
        }
    }
    // Normalize each sample
    if sum_a > 0.0 {
        let inv_a = 1.0 / sum_a;
        for x in local_a.iter_mut() {
            *x *= inv_a;
        }
    }
    if sum_b > 0.0 {
        let inv_b = 1.0 / sum_b;
        for x in local_b.iter_mut() {
            *x *= inv_b;
        }
    }

    // 5) partial sums => difference
    for (i, feat_name) in local_feats.iter().enumerate() {
        if let Some(&leaf_idx) = leaf_map.get(*feat_name) {
            let diff = local_a[i] - local_b[i];
            if diff.abs() > 1e-12 {
                partial_sums[leaf_idx] = diff;
                debug!(
                    "  => partial_sums[leaf_idx={}] = {} for feat='{}'",
                    leaf_idx, diff, feat_name
                );
            }
        }
    }

    // 6) Propagate partial sums up the chain
    let mut dist = 0.0;
    for i in 0..(num_nodes - 1) {
        let val = partial_sums[i];
        partial_sums[tint[i]] += val;
        dist += lint[i] * val.abs();
    }
    debug!("Final unweighted dist={}", dist);
    dist
}

//--------------------------------------------------------------------------------------//
//  Bitwise skip-zero logic for weighted
//--------------------------------------------------------------------------------------//

fn compute_unifrac_for_pair_weighted_bitwise(
    tint: &[usize],
    lint: &[f32],
    nodes_in_order: &[usize],
    leaf_map: &HashMap<String, usize>,
    feature_names: &[String],
    va: &[f32],
    vb: &[f32],
) -> f32 {
    debug!("=== compute_unifrac_for_pair_weighted_bitwise ===");
    let num_nodes = nodes_in_order.len();
    let mut partial_sums = vec![0.0; num_nodes];

    // 1) presence masks
    let mask_a = make_presence_mask_f32(va);
    let mask_b = make_presence_mask_f32(vb);

    // 2) combine
    let combined = or_masks(&mask_a, &mask_b);
    let non_zero_indices = extract_set_bits(&combined);
    debug!("non_zero_indices = {:?}", non_zero_indices);

    // 3) build local arrays
    let mut local_a = Vec::with_capacity(non_zero_indices.len());
    let mut local_b = Vec::with_capacity(non_zero_indices.len());
    let mut local_feats = Vec::with_capacity(non_zero_indices.len());
    for &idx in &non_zero_indices {
        local_a.push(va[idx]);
        local_b.push(vb[idx]);
        local_feats.push(&feature_names[idx]);
    }

    // 4) sum & normalize
    let sum_a: f32 = local_a.iter().sum();
    if sum_a > 0.0 {
        let inv_a = 1.0 / sum_a;
        for x in local_a.iter_mut() {
            *x *= inv_a;
        }
    }
    let sum_b: f32 = local_b.iter().sum();
    if sum_b > 0.0 {
        let inv_b = 1.0 / sum_b;
        for x in local_b.iter_mut() {
            *x *= inv_b;
        }
    }

    // 5) partial sums => diff
    for (i, feat_name) in local_feats.iter().enumerate() {
        if let Some(&leaf_idx) = leaf_map.get(*feat_name) {
            let diff = local_a[i] - local_b[i];
            if diff.abs() > 1e-12 {
                partial_sums[leaf_idx] = diff;
                debug!(
                    "   => partial_sums[leaf_idx={}] = {} for feat='{}'",
                    leaf_idx, diff, feat_name
                );
            }
        }
    }

    // 6) propagate partial sums
    let mut dist = 0.0;
    for i in 0..(num_nodes - 1) {
        let val = partial_sums[i];
        partial_sums[tint[i]] += val;
        dist += lint[i] * val.abs();
    }
    debug!("Final weighted dist={}", dist);
    dist
}

//=======================================================================================
//   Case of function pointers (cover Trait Fn , FnOnce ...)
// The book (Function item types):  " There is a coercion from function items to function pointers with the same signature  "
// The book (Call trait and coercions): "Non capturing closures can be coerced to function pointers with the same signature"

/// This type is for function with a C-API
/// Distances can be computed by such a function. It
/// takes as arguments the two (C, rust, julia) pointers to primitive type vectos and length
/// passed as a unsignedlonlong (64 bits) which is called c_ulonglong in Rust and Culonglong in Julia
///
type DistCFnPtr<T> = extern "C" fn(*const T, *const T, len: c_ulonglong) -> f32;

/// A structure to implement Distance Api for type DistCFnPtr\<T\>,
/// i.e distance provided by a C function pointer.  
/// It must be noted that this can be used in Julia via the macro @cfunction
/// to define interactiveley a distance function , compile it on the fly and sent it
/// to Rust via the init_hnsw_{f32, i32, u16, u32, u8} function
/// defined in libext
///
pub struct DistCFFI<T: Copy + Clone + Sized + Send + Sync> {
    dist_function: DistCFnPtr<T>,
}

impl<T: Copy + Clone + Sized + Send + Sync> DistCFFI<T> {
    pub fn new(f: DistCFnPtr<T>) -> Self {
        DistCFFI { dist_function: f }
    }
}

impl<T: Copy + Clone + Sized + Send + Sync> Distance<T> for DistCFFI<T> {
    fn eval(&self, va: &[T], vb: &[T]) -> f32 {
        // get pointers
        let len = va.len();
        let ptr_a = va.as_ptr();
        let ptr_b = vb.as_ptr();
        let dist = (self.dist_function)(ptr_a, ptr_b, len as c_ulonglong);
        log::trace!(
            "DistCFFI dist_function_ptr {:?} returning {:?} ",
            self.dist_function,
            dist
        );
        dist
    } // end of compute
} // end of impl block


//DistUniFrac_C
// Demonstration of calling `one_dense_pair_v2t` from Rust with tests
// -----------------------------------------------------------------------------
// 1) Reproduce some enums / types from C++ side
// -----------------------------------------------------------------------------
/// Mirror of your `compute_status` enum from C++.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ComputeStatus {
    Okay = 0,
    UnknownMethod = 1,
    TreeMissing = 2,
    TableMissing = 3,
    TableEmpty = 4,
    TableAndTreeDoNotOverlap = 5,
    OutputError = 6,
    InvalidMethod = 7,
    GroupingMissing = 8,
}

/// Opaque BPTree pointer type; the actual structure is in C++.
#[repr(C)]
pub struct OpaqueBPTree {
    _private: [u8; 0], // no fields in Rust
}

/// This struct holds parameters for one_dense_pair_v2t in C++.
#[repr(C)]
pub struct DistUniFrac_C {
    pub n_obs: c_uint,
    pub obs_ids: *const *const c_char,
    pub tree_data: *const OpaqueBPTree,
    pub unifrac_method: *const c_char,
    pub variance_adjust: bool,
    pub alpha: c_double,
    pub bypass_tips: bool,
}

// ----------------------------------------------------------------------------
// 2) Expose the extern "C" functions from your C++ library
// ----------------------------------------------------------------------------
extern "C" {
    /// Builds a BPTree from a Newick string.  
    /// On success, it writes an allocated `OpaqueBPTree*` into `tree_data_out`.
    pub fn load_bptree_opaque(
        newick: *const c_char,
        tree_data_out: *mut *mut OpaqueBPTree,
    );

    /// Frees the BPTree allocated by `load_bptree_opaque`.
    pub fn destroy_bptree_opaque(tree_data: *mut *mut OpaqueBPTree);

    /// The main distance function in C++:  
    ///   one_dense_pair_v2t(n_obs, obs_ids, sample1, sample2, tree_data, method_str, ...)
    pub fn one_dense_pair_v2t(
        n_obs: c_uint,
        obs_ids: *const *const c_char,
        sample1: *const c_double,
        sample2: *const c_double,
        tree_data: *const OpaqueBPTree,
        unifrac_method: *const c_char,
        variance_adjust: bool,
        alpha: c_double,
        bypass_tips: bool,
        result: *mut c_double,
    ) -> ComputeStatus;
}

// ----------------------------------------------------------------------------
// 3) Provide a function bridging from f32 slices to one_dense_pair_v2t
// ----------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn dist_unifrac_c(
    ctx: *const DistUniFrac_C,
    va: *const f32,
    vb: *const f32,
    length: c_ulonglong,
) -> f32 {
    if ctx.is_null() {
        eprintln!("dist_unifrac_c: NULL context pointer!");
        return 0.0;
    }
    let ctx_ref = unsafe { &*ctx };

    if length != ctx_ref.n_obs as u64 {
        eprintln!(
            "dist_unifrac_c: length mismatch. Got {}, expected {}",
            length, ctx_ref.n_obs
        );
        return 0.0;
    }

    let slice_a = unsafe { slice::from_raw_parts(va, length as usize) };
    let slice_b = unsafe { slice::from_raw_parts(vb, length as usize) };

    let mut buf_a = Vec::with_capacity(slice_a.len());
    let mut buf_b = Vec::with_capacity(slice_b.len());
    for (&a_val, &b_val) in slice_a.iter().zip(slice_b.iter()) {
        buf_a.push(a_val as f64);
        buf_b.push(b_val as f64);
    }

    let mut dist_out: c_double = 0.0;
    let status = unsafe {
        one_dense_pair_v2t(
            ctx_ref.n_obs,
            ctx_ref.obs_ids,
            buf_a.as_ptr(),
            buf_b.as_ptr(),
            ctx_ref.tree_data,
            ctx_ref.unifrac_method,
            ctx_ref.variance_adjust,
            ctx_ref.alpha,
            ctx_ref.bypass_tips,
            &mut dist_out,
        )
    };
    if status == ComputeStatus::Okay {
        dist_out as f32
    } else {
        eprintln!("one_dense_pair_v2t returned status {:?}", status);
        0.0
    }
}

// ----------------------------------------------------------------------------
// 4) Provide create/destroy for DistUniFrac_C
// ----------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn dist_unifrac_create(
    n_obs: c_uint,
    obs_ids: *const *const c_char,
    tree_data: *const OpaqueBPTree,
    unifrac_method: *const c_char,
    variance_adjust: bool,
    alpha: c_double,
    bypass_tips: bool,
) -> *mut DistUniFrac_C {
    let ctx = DistUniFrac_C {
        n_obs,
        obs_ids,
        tree_data,
        unifrac_method,
        variance_adjust,
        alpha,
        bypass_tips,
    };
    Box::into_raw(Box::new(ctx))
}

#[no_mangle]
pub extern "C" fn dist_unifrac_destroy(ctx_ptr: *mut DistUniFrac_C) {
    if !ctx_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ctx_ptr);
        }
    }
}

// ----------------------------------------------------------------------------
// 5) DistUniFracCFFI struct implementing Distance<f32>
// ----------------------------------------------------------------------------

#[derive(Clone)]  // We'll do a shallow clone of the pointer
pub struct DistUniFracCFFI {
    ctx: *mut DistUniFrac_C,
    func: extern "C" fn(*const DistUniFrac_C, *const f32, *const f32, c_ulonglong) -> f32,
}

impl DistUniFracCFFI {
    pub fn new(ctx: *mut DistUniFrac_C) -> Self {
        DistUniFracCFFI {
            ctx,
            func: dist_unifrac_c,
        }
    }

    /// A direct convenience method
    pub fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        (self.func)(
            self.ctx,
            va.as_ptr(),
            vb.as_ptr(),
            va.len() as c_ulonglong,
        )
    }
}

/// Implement the `Distance<f32>` trait so that DistUniFracCFFI can be used in HNSW:
impl Distance<f32> for DistUniFracCFFI {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        self.eval(va, vb)
    }
}

unsafe impl Send for DistUniFracCFFI {}
unsafe impl Sync for DistUniFracCFFI {}

///end DistUniFrac_C

//========================================================================================================

/// This structure is to let user define their own distance with closures.
pub struct DistFn<T: Copy + Clone + Sized + Send + Sync> {
    dist_function: Box<dyn Fn(&[T], &[T]) -> f32 + Send + Sync>,
}

impl<T: Copy + Clone + Sized + Send + Sync> DistFn<T> {
    /// construction of a DistFn
    pub fn new(f: Box<dyn Fn(&[T], &[T]) -> f32 + Send + Sync>) -> Self {
        DistFn { dist_function: f }
    }
}

impl<T: Copy + Clone + Sized + Send + Sync> Distance<T> for DistFn<T> {
    fn eval(&self, va: &[T], vb: &[T]) -> f32 {
        (self.dist_function)(va, vb)
    }
}

//=======================================================================================

/// This structure uses a Rust function pointer to define the distance.
/// For commodity it can build upon a fonction returning a f64.
/// Beware that if F is f64, the distance converted to f32 can overflow!

#[derive(Copy, Clone)]
pub struct DistPtr<T: Copy + Clone + Sized + Send + Sync, F: Float> {
    dist_function: fn(&[T], &[T]) -> F,
}

impl<T: Copy + Clone + Sized + Send + Sync, F: Float> DistPtr<T, F> {
    /// construction of a DistPtr
    pub fn new(f: fn(&[T], &[T]) -> F) -> Self {
        DistPtr { dist_function: f }
    }
}

/// beware that if F is f64, the distance converted to f32 can overflow!
impl<T: Copy + Clone + Sized + Send + Sync, F: Float> Distance<T> for DistPtr<T, F> {
    fn eval(&self, va: &[T], vb: &[T]) -> f32 {
        (self.dist_function)(va, vb).to_f32().unwrap()
    }
}

//=======================================================================================

#[cfg(test)]

mod tests {
    use super::*;
    use log::debug;
    use env_logger::Env;
    use std::ffi::{CString};
    use std::ptr;

    fn init_log() -> u64 {
        let mut builder = env_logger::Builder::from_default_env();
        let _ = builder.is_test(true).try_init();
        println!("\n ************** initializing logger *****************\n");
        return 1;
    }
    /// Helper to create a raw array of `*const c_char` for T1..T6.
    fn make_obs_ids() -> Vec<*mut c_char> {
        let obs = vec![
            CString::new("T1").unwrap(),
            CString::new("T2").unwrap(),
            CString::new("T3").unwrap(),
            CString::new("T4").unwrap(),
            CString::new("T5").unwrap(),
            CString::new("T6").unwrap(),
        ];
        let mut c_ptrs = Vec::with_capacity(obs.len());
        for s in obs {
            // into_raw() -> *mut c_char
            c_ptrs.push(s.into_raw()); 
        }
        c_ptrs
    }
    
    fn free_obs_ids(c_ptrs: &mut [*mut c_char]) {
        for &mut ptr in c_ptrs {
            if !ptr.is_null() {
                // Convert back to a CString so it will be freed
                unsafe {
                    let _ = CString::from_raw(ptr);
                }
            }
        }
    }

    #[test]
    fn test_access_to_dist_l1() {
        let distl1 = DistL1;
        //
        let v1: Vec<i32> = vec![1, 2, 3];
        let v2: Vec<i32> = vec![2, 2, 3];

        let d1 = Distance::eval(&distl1, &v1, &v2);
        assert_eq!(d1, 1 as f32);

        let v3: Vec<f32> = vec![1., 2., 3.];
        let v4: Vec<f32> = vec![2., 2., 3.];
        let d2 = distl1.eval(&v3, &v4);
        assert_eq!(d2, 1 as f32);
    }

    #[test]
    fn have_avx2() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                println!("I have avx2");
            } else {
                println!(" ************ I DO NOT  have avx2  ***************");
            }
        }
    } // end if

    #[test]
    fn have_avx512f() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") {
                println!("I have avx512f");
            } else {
                println!(" ************ I DO NOT  have avx512f  ***************");
            }
        } // end of have_avx512f
    }

    #[test]
    fn have_sse2() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse2") {
                println!("I have sse2");
            } else {
                println!(" ************ I DO NOT  have SSE2  ***************");
            }
        }
    } // end of have_sse2

    #[test]
    fn test_access_to_dist_cos() {
        let distcos = DistCosine;
        //
        let v1: Vec<i32> = vec![1, -1, 1];
        let v2: Vec<i32> = vec![2, 1, -1];

        let d1 = Distance::eval(&distcos, &v1, &v2);
        assert_eq!(d1, 1. as f32);
        //
        let v1: Vec<f32> = vec![1.234, -1.678, 1.367];
        let v2: Vec<f32> = vec![4.234, -6.678, 10.367];
        let d1 = Distance::eval(&distcos, &v1, &v2);

        let mut normv1 = 0.;
        let mut normv2 = 0.;
        let mut prod = 0.;
        for i in 0..v1.len() {
            prod += v1[i] * v2[i];
            normv1 += v1[i] * v1[i];
            normv2 += v2[i] * v2[i];
        }
        let dcos = 1. - prod / (normv1 * normv2).sqrt();
        println!("dist cos avec macro = {:?} ,  avec for {:?}", d1, dcos);
    }

    #[test]
    fn test_dot_distances() {
        let mut v1: Vec<f32> = vec![1.234, -1.678, 1.367];
        let mut v2: Vec<f32> = vec![4.234, -6.678, 10.367];

        let mut normv1 = 0.;
        let mut normv2 = 0.;
        let mut prod = 0.;
        for i in 0..v1.len() {
            prod += v1[i] * v2[i];
            normv1 += v1[i] * v1[i];
            normv2 += v2[i] * v2[i];
        }
        let dcos = 1. - prod / (normv1 * normv2).sqrt();
        //
        l2_normalize(&mut v1);
        l2_normalize(&mut v2);

        println!(" after normalisation v1 = {:?}", v1);
        let dot = DistDot.eval(&v1, &v2);
        println!(
            "dot  cos avec prenormalisation  = {:?} ,  avec for {:?}",
            dot, dcos
        );
    }

    #[test]
    fn test_l1() {
        init_log();
        //
        let va: Vec<f32> = vec![1.234, -1.678, 1.367, 1.234, -1.678, 1.367];
        let vb: Vec<f32> = vec![4.234, -6.678, 10.367, 1.234, -1.678, 1.367];
        //
        let dist = DistL1.eval(&va, &vb);
        let dist_check = va
            .iter()
            .zip(vb.iter())
            .map(|t| (*t.0 as f32 - *t.1 as f32).abs())
            .sum::<f32>();
        //
        log::info!(" dist : {:.5e} dist_check : {:.5e}", dist, dist_check);
        assert!((dist - dist_check).abs() / dist_check < 1.0e-5);
    } // end of test_l1

    #[test]
    fn test_jaccard_u16() {
        let v1: Vec<u16> = vec![1, 2, 1, 4, 3];
        let v2: Vec<u16> = vec![2, 2, 1, 5, 6];

        let dist = DistJaccard.eval(&v1, &v2);
        println!("dist jaccard = {:?}", dist);
        assert_eq!(dist, 1. - 11. / 16.);
    } // end of test_jaccard

    #[test]
    fn test_levenshtein() {
        let mut v1: Vec<u16> = vec![1, 2, 3, 4];
        let mut v2: Vec<u16> = vec![1, 2, 3, 3];
        let mut dist = DistLevenshtein.eval(&v1, &v2);
        println!("dist levenshtein = {:?}", dist);
        assert_eq!(dist, 1.0);
        v1 = vec![1, 2, 3, 4];
        v2 = vec![1, 2, 3, 4];
        dist = DistLevenshtein.eval(&v1, &v2);
        println!("dist levenshtein = {:?}", dist);
        assert_eq!(dist, 0.0);
        v1 = vec![1, 1, 1, 4];
        v2 = vec![1, 2, 3, 4];
        dist = DistLevenshtein.eval(&v1, &v2);
        println!("dist levenshtein = {:?}", dist);
        assert_eq!(dist, 2.0);
        v2 = vec![1, 1, 1, 4];
        v1 = vec![1, 2, 3, 4];
        dist = DistLevenshtein.eval(&v1, &v2);
        println!("dist levenshtein = {:?}", dist);
        assert_eq!(dist, 2.0);
    } // end of test_levenshtein

    extern "C" fn dist_func_float(va: *const f32, vb: *const f32, len: c_ulonglong) -> f32 {
        let mut dist: f32 = 0.;
        let sa = unsafe { std::slice::from_raw_parts(va, len as usize) };
        let sb = unsafe { std::slice::from_raw_parts(vb, len as usize) };

        for i in 0..len {
            dist += (sa[i as usize] - sb[i as usize]).abs().sqrt();
        }
        dist
    }

    #[test]
    fn test_dist_ext_float() {
        let va: Vec<f32> = vec![1., 2., 3.];
        let vb: Vec<f32> = vec![1., 2., 3.];
        println!("in test_dist_ext_float");
        let dist1 = dist_func_float(va.as_ptr(), vb.as_ptr(), va.len() as c_ulonglong);
        println!("test_dist_ext_float computed : {:?}", dist1);

        let mydist = DistCFFI::<f32>::new(dist_func_float);

        let dist2 = mydist.eval(&va, &vb);
        assert_eq!(dist1, dist2);
    } // end test_dist_ext_float

    #[test]

    fn test_my_closure() {
        //        use hnsw_rs::dist::Distance;
        let weight = vec![0.1, 0.8, 0.1];
        let my_fn = move |va: &[f32], vb: &[f32]| -> f32 {
            // should check that we work with same size for va, vb, and weight...
            let mut dist: f32 = 0.;
            for i in 0..va.len() {
                dist += weight[i] * (va[i] - vb[i]).abs();
            }
            dist
        };
        let my_boxed_f = Box::new(my_fn);
        let my_boxed_dist = DistFn::<f32>::new(my_boxed_f);
        let va: Vec<f32> = vec![1., 2., 3.];
        let vb: Vec<f32> = vec![2., 2., 4.];
        let dist = my_boxed_dist.eval(&va, &vb);
        println!("test_my_closure computed : {:?}", dist);
        // try allocation Hnsw
        //        let _hnsw = Hnsw::<f32, hnsw_rs::dist::DistFn<f32>>::new(10, 3, 100, 16, my_boxed_dist);
        //
        assert_eq!(dist, 0.2);
    } // end of test_my_closure

    #[test]
    fn test_hellinger() {
        let length = 9;
        let mut p_data = Vec::with_capacity(length);
        let mut q_data = Vec::with_capacity(length);
        for _ in 0..length {
            p_data.push(1. / length as f32);
            q_data.push(1. / length as f32);
        }
        p_data[0] -= 1. / (2 * length) as f32;
        p_data[1] += 1. / (2 * length) as f32;
        //
        let dist = DistHellinger.eval(&p_data, &q_data);

        let dist_exact_fn = |n: usize| -> f32 {
            let d1 = (4. - (6 as f32).sqrt() - (2 as f32).sqrt()) / n as f32;
            d1.sqrt() / (2 as f32).sqrt()
        };
        let dist_exact = dist_exact_fn(length);
        //
        log::info!("dist computed {:?} dist exact{:?} ", dist, dist_exact);
        println!("dist computed  {:?} , dist exact {:?} ", dist, dist_exact);
        //
        assert!((dist - dist_exact).abs() < 1.0e-5);
    }

    #[test]
    fn test_jeffreys() {
        // this essentially test av2 implementation for f32
        let length = 19;
        let mut p_data: Vec<f32> = Vec::with_capacity(length);
        let mut q_data: Vec<f32> = Vec::with_capacity(length);
        for _ in 0..length {
            p_data.push(1. / length as f32);
            q_data.push(1. / length as f32);
        }
        p_data[0] -= 1. / (2 * length) as f32;
        p_data[1] += 1. / (2 * length) as f32;
        q_data[10] += 1. / (2 * length) as f32;
        //
        let dist_eval = DistJeffreys.eval(&p_data, &q_data);
        let mut dist_test = 0.;
        for i in 0..length {
            dist_test +=
                (p_data[i] - q_data[i]) * (p_data[i].max(M_MIN) / q_data[i].max(M_MIN)).ln();
        }
        //
        log::info!("dist eval {:?} dist test{:?} ", dist_eval, dist_test);
        println!("dist eval  {:?} , dist test {:?} ", dist_eval, dist_test);
        assert!(dist_test >= 0.);
        assert!((dist_eval - dist_test).abs() < 1.0e-5);
    }

    #[test]
    fn test_jensenshannon() {
        init_log();
        //
        let length = 19;
        let mut p_data: Vec<f32> = Vec::with_capacity(length);
        let mut q_data: Vec<f32> = Vec::with_capacity(length);
        for _ in 0..length {
            p_data.push(1. / length as f32);
            q_data.push(1. / length as f32);
        }
        p_data[0] -= 1. / (2 * length) as f32;
        p_data[1] += 1. / (2 * length) as f32;
        q_data[10] += 1. / (2 * length) as f32;
        p_data[12] = 0.;
        q_data[12] = 0.;
        //
        let dist_eval = DistJensenShannon.eval(&p_data, &q_data);
        //
        log::info!("dist eval {:?} ", dist_eval);
        println!("dist eval  {:?} ", dist_eval);
    }

    #[allow(unused)]
    use rand::distributions::{Distribution, Uniform};

    // to be run with and without simdeez_f
    #[test]
    fn test_hamming_f64() {
        init_log();

        let size_test = 500;
        let fmax: f64 = 3.;
        let mut rng = rand::thread_rng();
        for i in 300..size_test {
            // generer 2 va et vb s des vecteurs<i32> de taille i  avec des valeurs entre -imax et + imax et controler les resultat
            let between = Uniform::<f64>::from(-fmax..fmax);
            let va: Vec<f64> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            let mut vb: Vec<f64> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            // reset half of vb to va
            for i in 0..i / 2 {
                vb[i] = va[i];
            }

            let easy_dist: u32 = va
                .iter()
                .zip(vb.iter())
                .map(|(a, b)| if a != b { 1 } else { 0 })
                .sum();
            let h_dist = DistHamming.eval(&va, &vb);
            let easy_dist = easy_dist as f32 / va.len() as f32;
            let j_exact = ((i / 2) as f32) / (i as f32);
            log::debug!(
                "test size {:?}  HammingDist {:.3e} easy : {:.3e} exact : {:.3e} ",
                i,
                h_dist,
                easy_dist,
                j_exact
            );
            if (easy_dist - h_dist).abs() > 1.0e-5 {
                println!(" jhamming = {:?} , jexact = {:?}", h_dist, easy_dist);
                log::debug!("va = {:?}", va);
                log::debug!("vb = {:?}", vb);
                std::process::exit(1);
            }
            if (j_exact - h_dist).abs() > 1. / i as f32 + 1.0E-5 {
                println!(
                    " jhamming = {:?} , jexact = {:?}, j_easy : {:?}",
                    h_dist, j_exact, easy_dist
                );
                log::debug!("va = {:?}", va);
                log::debug!("vb = {:?}", vb);
                std::process::exit(1);
            }
        }
    } // end of test_hamming_f64

    #[test]
    fn test_hamming_f32() {
        init_log();

        let size_test = 500;
        let fmax: f32 = 3.;
        let mut rng = rand::thread_rng();
        for i in 300..size_test {
            // generer 2 va et vb s des vecteurs<i32> de taille i  avec des valeurs entre -imax et + imax et controler les resultat
            let between = Uniform::<f32>::from(-fmax..fmax);
            let va: Vec<f32> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            let mut vb: Vec<f32> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            // reset half of vb to va
            for i in 0..i / 2 {
                vb[i] = va[i];
            }

            let easy_dist: u32 = va
                .iter()
                .zip(vb.iter())
                .map(|(a, b)| if a != b { 1 } else { 0 })
                .sum();
            let h_dist = DistHamming.eval(&va, &vb);
            let easy_dist = easy_dist as f32 / va.len() as f32;
            let j_exact = ((i / 2) as f32) / (i as f32);
            log::debug!(
                "test size {:?}  HammingDist {:.3e} easy : {:.3e} exact : {:.3e} ",
                i,
                h_dist,
                easy_dist,
                j_exact
            );
            if (easy_dist - h_dist).abs() > 1.0e-5 {
                println!(
                    " jhamming = {:?} , jexact = {:?}, j_easy : {:?}",
                    h_dist, j_exact, easy_dist
                );
                log::debug!("va = {:?}", va);
                log::debug!("vb = {:?}", vb);
                std::process::exit(1);
            }
            if (j_exact - h_dist).abs() > 1. / i as f32 + 1.0E-5 {
                println!(
                    " jhamming = {:?} , jexact = {:?}, j_easy : {:?}",
                    h_dist, j_exact, easy_dist
                );
                log::debug!("va = {:?}", va);
                log::debug!("vb = {:?}", vb);
                std::process::exit(1);
            }
        }
    } // end of test_hamming_f32

    #[cfg(feature = "stdsimd")]
    #[test]
    fn test_feature_simd() {
        init_log();
        log::info!("I have activated stdsimd");
    } // end of test_feature_simd

    #[test]
    #[cfg(feature = "simdeez_f")]
    fn test_feature_simdeez() {
        init_log();
        log::info!("I have activated simdeez");
    } // end of test_feature_simd

    /// Example test using unweighted EMDUniFrac with presence/absence
    #[test]
    fn test_unifrac_unweighted() {
        let _ = env_logger::Builder::from_env(Env::default().default_filter_or("debug"))
        .is_test(true)
        .try_init();

        let newick_str = "((T1:0.1,(T2:0.05,T3:0.05):0.02):0.3,(T4:0.2,(T5:0.1,T6:0.15):0.05):0.4);";
        let feature_names = vec!["T1","T2","T3","T4","T5","T6"]
            .into_iter().map(|s| s.to_string()).collect();
        let dist_uni = DistUniFrac::new(newick_str, false, feature_names).unwrap();

        // SampleA: T1=7, T3=5, T4=2 => presence
        // SampleB: T2=3, T5=4, T6=9 => presence
        let va = vec![7.0, 0.0, 5.0, 2.0, 0.0, 0.0];
        let vb = vec![0.0, 3.0, 0.0, 0.0, 0.0, 9.0];

        let d = dist_uni.eval(&va, &vb);
        println!("Unweighted EMDUniFrac(A,B) = {}", d);
        // Should be ~0.4833
        // e.g. assert!((d - 0.4833).abs() < 1e-4);
    }

    #[test]
    fn test_unifrac_weighted() {
        let _ = env_logger::Builder::from_env(Env::default().default_filter_or("debug"))
        .is_test(true)
        .try_init();
        let newick_str = "((T1:0.1,(T2:0.05,T3:0.05):0.02):0.3,(T4:0.2,(T5:0.1,T6:0.15):0.05):0.4);";
        let feature_names = vec!["T1","T2","T3","T4","T5","T6"]
            .into_iter().map(|s| s.to_string()).collect();
        let dist_uni = DistUniFrac::new(newick_str, true, feature_names).unwrap();

        // Weighted with same data
        let va = vec![7.0, 0.0, 5.0, 2.0, 0.0, 0.0];
        let vb = vec![0.0, 3.0, 0.0, 0.0, 0.0, 9.0];

        let d = dist_uni.eval(&va, &vb);
        println!("Weighted EMDUniFrac(A,B) = {}", d);
        // Should be ~0.7279
        // e.g. assert!((d - 0.7279).abs() < 1e-4);
    }

    #[test]
    fn test_unifrac_unweighted_c_api() {
        // 1) Build a BPTree from a Newick string
        let newick_str = CString::new("((T1:0.1,(T2:0.05,T3:0.05):0.02):0.3,(T4:0.2,(T5:0.1,T6:0.15):0.05):0.4);").unwrap();
        let mut tree_ptr: *mut OpaqueBPTree = ptr::null_mut();
        unsafe {
            load_bptree_opaque(newick_str.as_ptr(), &mut tree_ptr);
        }
        assert!(!tree_ptr.is_null(), "Failed to build tree");

        // 2) Create obs_ids = T1..T6

        let mut obs_ids = make_obs_ids();
        let obs_ids_ptr = obs_ids.as_ptr() as *const *const c_char;

        // 3) Build DistUniFrac_C context for "unweighted"
        let method_str = CString::new("unweighted").unwrap();
        let ctx_ptr = dist_unifrac_create(
            6, // n_obs
            obs_ids_ptr,
            tree_ptr,
            method_str.as_ptr(),
            false,   // variance_adjust = false
            1.0,     // alpha
            false,   // bypass_tips = false
        );
        assert!(!ctx_ptr.is_null());

        // 4) Use dist_unifrac_c or DistUniFracCFFI to compute distance
        let dist_obj = DistUniFracCFFI::new(ctx_ptr);

        // Same example data:
        // Sample A => T1=7, T3=5, T4=2 => presence
        // Sample B => T2=3, T5=4, T6=9 => presence
        // We'll put them in f32 arrays of length 6: [T1,T2,T3,T4,T5,T6].
        let va = [7.0, 0.0, 5.0, 2.0, 0.0, 0.0];
        let vb = [0.0, 3.0, 0.0, 0.0, 4.0, 9.0];

        let dist = dist_obj.eval(&va, &vb);
        println!("Unweighted UniFrac(A,B) = {}", dist);
        // You mention it should be ~0.4833. Let's check tolerance
        // 5) Clean up
        unsafe {
            dist_unifrac_destroy(ctx_ptr);
            destroy_bptree_opaque(&mut tree_ptr);
        }
        free_obs_ids(&mut obs_ids);
    }

    #[test]
    fn test_unifrac_weighted_c_api() {
        // 1) Build the same BPTree
        let newick_str = CString::new("((T1:0.1,(T2:0.05,T3:0.05):0.02):0.3,(T4:0.2,(T5:0.1,T6:0.15):0.05):0.4);").unwrap();
        let mut tree_ptr: *mut OpaqueBPTree = ptr::null_mut();
        unsafe {
            load_bptree_opaque(newick_str.as_ptr(), &mut tree_ptr);
        }
        assert!(!tree_ptr.is_null(), "Failed to build tree");

        // 2) obs_ids
        let mut obs_ids = make_obs_ids();
        let obs_ids_ptr = obs_ids.as_ptr() as *const *const c_char;

        // 3) DistUniFrac_C for "weighted_normalized" (or just "weighted" if your C++ wants that)
        let method_str = CString::new("weighted_normalized").unwrap();
        let ctx_ptr = dist_unifrac_create(
            6,
            obs_ids_ptr,
            tree_ptr,
            method_str.as_ptr(),
            false, // variance_adjust
            1.0,   // alpha
            false, // bypass_tips
        );
        assert!(!ctx_ptr.is_null());

        // 4) Evaluate
        let dist_obj = DistUniFracCFFI::new(ctx_ptr);
        let va = [7.0, 0.0, 5.0, 2.0, 0.0, 0.0];
        let vb = [0.0, 3.0, 0.0, 0.0, 4.0, 9.0];

        let dist = dist_obj.eval(&va, &vb);
        println!("Weighted UniFrac(A,B) = {}", dist);


        // 5) Cleanup
        unsafe {
            dist_unifrac_destroy(ctx_ptr);
            destroy_bptree_opaque(&mut tree_ptr);
        }
        free_obs_ids(&mut obs_ids);
    }
} // end of module tests
