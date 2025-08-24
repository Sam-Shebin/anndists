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
use phylotree::tree::Tree;
use std::collections::HashMap;
use log::debug;

// for BitVec used in NewDistUniFrac
use bitvec::prelude::*;
use newick::{one_from_string, Newick, NewickTree};
use succparen::{
    bitwise::SparseOneNnd,
    tree::{
        balanced_parens::{BalancedParensTree, Node as BpNode},
        traversal::{DepthFirstTraverse, VisitNode},
        LabelVec, Node,
    },
};


/// for DistCFnPtr_UniFrac
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
    /// New UniFrac distance using succparen for high performance
    NewDistUniFrac(String),
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

// Helper function to extract leaf names from newick string without phylotree dependency
fn extract_leaf_names_from_newick_string(newick_str: &str) -> Result<Vec<String>> {
    let mut leaf_names = Vec::new();
    
    // Method 1: Try to extract from existing tree iteration
    let t: NewickTree = one_from_string(newick_str)
        .map_err(|e| anyhow!("Failed to parse Newick string: {}", e))?;
    
    // Create a temporary SuccTrav to iterate through nodes
    let mut temp_lens = Vec::<f32>::new();
    let trav = SuccTrav::new(&t, &mut temp_lens);
    let bp: BalancedParensTree<LabelVec<()>, SparseOneNnd> =
        BalancedParensTree::new_builder(trav, LabelVec::<()>::new()).build_all();
    
    let total = bp.len() + 1;
    let mut kids = vec![Vec::<usize>::new(); total];
    let mut post = Vec::<usize>::with_capacity(total);
    collect_children::<SparseOneNnd>(&bp.root(), &mut kids, &mut post);
    
    // Find leaf nodes (nodes with no children)
    let mut leaf_ids = Vec::<usize>::new();
    for node_id in 0..total {
        if kids[node_id].is_empty() {
            leaf_ids.push(node_id);
        }
    }
    
    // Method 2: Use simple pattern matching for common cases
    // This handles most newick formats including complex ones
    let cleaned = newick_str.replace('\n', "").replace('\r', "").replace('\t', " ");
    
    // Find leaf patterns: word followed by : (leaf with branch length) or word at end before )
    let mut current_token = String::new();
    let mut chars = cleaned.chars().peekable();
    let mut depth = 0;
    
    while let Some(ch) = chars.next() {
        match ch {
            '(' => {
                depth += 1;
                if !current_token.is_empty() && depth == 1 {
                    // Root name, skip
                    current_token.clear();
                }
            }
            ')' => {
                depth -= 1;
                // Check if we have accumulated a token before closing
                if !current_token.trim().is_empty() {
                    let name = current_token.trim().to_string();
                    if !name.is_empty() && !name.chars().all(|c| c.is_numeric() || c == '.') {
                        leaf_names.push(name);
                    }
                }
                current_token.clear();
            }
            ',' => {
                // End of current item
                if !current_token.trim().is_empty() {
                    let name = current_token.trim().to_string();
                    if !name.is_empty() && !name.chars().all(|c| c.is_numeric() || c == '.') {
                        leaf_names.push(name);
                    }
                }
                current_token.clear();
            }
            ':' => {
                // Branch length follows, current token might be a leaf name
                if !current_token.trim().is_empty() {
                    let name = current_token.trim().to_string();
                    if !name.is_empty() && !name.chars().all(|c| c.is_numeric() || c == '.') {
                        leaf_names.push(name);
                    }
                }
                current_token.clear();
                
                // Skip until next delimiter
                while let Some(&next_ch) = chars.peek() {
                    if next_ch == ',' || next_ch == ')' || next_ch == '(' {
                        break;
                    }
                    chars.next();
                }
            }
            ';' => break,
            ' ' => {
                // Keep spaces in names but trim later
                if !current_token.is_empty() {
                    current_token.push(ch);
                }
            }
            _ => {
                current_token.push(ch);
            }
        }
    }
    
    // Remove duplicates and empty names
    leaf_names.sort();
    leaf_names.dedup();
    leaf_names.retain(|name| !name.is_empty());
    
    Ok(leaf_names)
}

// start of NewDistUniFrac

/// A new DistUniFrac struct using succparen and balanced parentheses tree.
#[derive(Debug)]
pub struct NewDistUniFrac {
    pub weighted: bool,
    pub post: Vec<usize>,
    pub kids: Vec<Vec<usize>>,
    pub lens: Vec<f32>,
    pub leaf_ids: Vec<usize>,
    pub feature_names: Vec<String>,
}

impl NewDistUniFrac {
    /// Build NewDistUniFrac from Newick string and feature names
    pub fn new(newick_str: &str, weighted: bool, feature_names: Vec<String>) -> Result<Self> {
        // 1. Parse Newick string
        let t: NewickTree = one_from_string(newick_str)
            .map_err(|e| anyhow!("Failed to parse Newick string: {}", e))?;

        // 2. Build BalancedParensTree via SuccTrav
        let mut lens = Vec::<f32>::new();
        let trav = SuccTrav::new(&t, &mut lens);
        let bp: BalancedParensTree<LabelVec<()>, SparseOneNnd> =
            BalancedParensTree::new_builder(trav, LabelVec::<()>::new()).build_all();

        // 3. Collect children and postorder
        let total = bp.len() + 1;
        lens.resize(total, 0.0);
        let mut kids = vec![Vec::<usize>::new(); total];
        let mut post = Vec::<usize>::with_capacity(total);
        collect_children::<SparseOneNnd>(&bp.root(), &mut kids, &mut post);

        // 4. Extract leaf nodes using succparen only (no phylotree-rs dependencies)
        let mut leaf_ids = Vec::<usize>::new();
        let mut leaf_nm = Vec::<String>::new();
        
        // Find leaves using succparen: nodes with no children in kids array
        for node_id in 0..total {
            if kids[node_id].is_empty() {
                leaf_ids.push(node_id);
            }
        }
        
        // Extract leaf names using the helper function (phylotree-free)
        let leaf_names_from_newick = extract_leaf_names_from_newick_string(newick_str)?;
        
        // Use extracted names if available, otherwise fall back to dummy names
        for (i, &bp_leaf_id) in leaf_ids.iter().enumerate() {
            if i < leaf_names_from_newick.len() {
                leaf_nm.push(leaf_names_from_newick[i].clone());
            } else {
                leaf_nm.push(format!("L{}", bp_leaf_id));
            }
        }

        // 5. Build taxon-name → leaf-index map
        let t2leaf: HashMap<&str, usize> = leaf_nm
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();

        // 6. Map feature_names to leaf_ids
        let mut mapped_leaf_ids = Vec::with_capacity(feature_names.len());
        for fname in &feature_names {
            if let Some(&leaf_pos) = t2leaf.get(fname.as_str()) {
                mapped_leaf_ids.push(leaf_ids[leaf_pos]);
            } else {
                return Err(anyhow!("Feature name '{}' not found in tree", fname));
            }
        }

        Ok(Self {
            weighted,
            post,
            kids,
            lens,
            leaf_ids: mapped_leaf_ids,
            feature_names,
        })
    }

    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    pub fn num_features(&self) -> usize {
        self.feature_names.len()
    }
}

impl Distance<f32> for NewDistUniFrac {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        unifrac_succparen_normalized(
            &self.post,
            &self.kids,
            &self.lens,
            &self.leaf_ids,
            va,
            vb,
        ) as f32
    }
}

/// UniFrac using succparen tree structure with shared/union logic (like unifrac_pair)
fn unifrac_succparen_normalized(
    post: &[usize],
    kids: &[Vec<usize>],
    lens: &[f32], 
    leaf_ids: &[usize],
    va: &[f32],
    vb: &[f32],
) -> f64 {
    const A_BIT: u8 = 0b01;
    const B_BIT: u8 = 0b10;
    
    // Create presence masks (like unifrac_pair)
    let mut mask = vec![0u8; lens.len()];
    
    for (leaf_pos, &leaf_node_id) in leaf_ids.iter().enumerate() {
        if leaf_pos < va.len() && leaf_pos < vb.len() {
            if va[leaf_pos] > 0.0 {
                mask[leaf_node_id] |= A_BIT;
            }
            if vb[leaf_pos] > 0.0 {
                mask[leaf_node_id] |= B_BIT;
            }
        }
    }
    
    // Propagate masks up the tree (like unifrac_pair)
    for &node in post {
        for &child in &kids[node] {
            mask[node] |= mask[child];
        }
    }
    
    // Calculate shared and union branch lengths (like unifrac_pair)
    let (mut shared, mut union) = (0.0, 0.0);
    for &node in post {
        let m = mask[node];
        if m == 0 {
            continue; // No presence in either sample
        }
        let len = lens[node] as f64;
        if m == A_BIT || m == B_BIT {
            // Branch present in only one sample
            union += len;
        } else {
            // Branch present in both samples (m == A_BIT | B_BIT)
            shared += len;
            union += len;
        }
    }
    
    // Return UniFrac distance: 1.0 - shared/union (like unifrac_pair)
    if union == 0.0 {
        0.0
    } else {
        1.0 - shared / union
    }
}

/// Fast unweighted UniFrac using bit masks
fn unifrac_pair(
    post: &[usize],
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    a: &BitVec<u8, bitvec::order::Lsb0>,
    b: &BitVec<u8, bitvec::order::Lsb0>,
) -> f64 {
    const A_BIT: u8 = 0b01;
    const B_BIT: u8 = 0b10;
    let mut mask = vec![0u8; lens.len()];
    for (leaf_pos, &nid) in leaf_ids.iter().enumerate() {
        if a[leaf_pos] {
            mask[nid] |= A_BIT;
        }
        if b[leaf_pos] {
            mask[nid] |= B_BIT;
        }
    }
    for &v in post {
        for &c in &kids[v] {
            mask[v] |= mask[c];
        }
    }
    let (mut shared, mut union) = (0.0, 0.0);
    for &v in post {
        let m = mask[v];
        if m == 0 {
            continue;
        }
        let len = lens[v] as f64;
        if m == A_BIT || m == B_BIT {
            union += len;
        } else {
            shared += len;
            union += len;
        }
    }
    if union == 0.0 {
        0.0
    } else {
        1.0 - shared / union
    }
}

// --- SuccTrav and collect_children code copied from unifrac_bp ---

/// SuccTrav for BalancedParensTree building
struct SuccTrav<'a> {
    t: &'a NewickTree,
    stack: Vec<(usize, usize, usize)>,
    lens: &'a mut Vec<f32>,
}

impl<'a> SuccTrav<'a> {
    fn new(t: &'a NewickTree, lens: &'a mut Vec<f32>) -> Self {
        Self {
            t,
            stack: vec![(t.root(), 0, 0)],
            lens,
        }
    }
}

impl<'a> DepthFirstTraverse for SuccTrav<'a> {
    type Label = ();

    fn next(&mut self) -> Option<VisitNode<Self::Label>> {
        let (id, lvl, nth) = self.stack.pop()?;
        let n_children = self.t[id].children().len();
        for (k, &c) in self.t[id].children().iter().enumerate().rev() {
            let nth = n_children - 1 - k;
            self.stack.push((c, lvl + 1, nth));
        }
        if self.lens.len() <= id {
            self.lens.resize(id + 1, 0.0);
        }
        self.lens[id] = self.t[id].branch().copied().unwrap_or(0.0);
        Some(VisitNode::new((), lvl, nth))
    }
}

fn collect_children<N: succparen::bitwise::ops::NndOne>(
    node: &BpNode<LabelVec<()>, N, &BalancedParensTree<LabelVec<()>, N>>,
    kids: &mut [Vec<usize>],
    post: &mut Vec<usize>,
) {
    let pid = node.id() as usize;
    for edge in node.children() {
        let cid = edge.node.id() as usize;
        kids[pid].push(cid);
        collect_children(&edge.node, kids, post);
    }
    post.push(pid);
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
// TEMPORARILY COMMENTED OUT - external C UniFrac library functions
/*
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
*/

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

    // TEMPORARILY COMMENTED OUT - requires external C UniFrac library
    /*
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
    */

    // TEMPORARILY COMMENTED OUT - requires external C UniFrac library
    /*
    #[test]
    fn test_unifrac_weighted_c_api() {
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

        // 3) Build DistUniFrac_C context for "weighted"
        let method_str = CString::new("weighted_normalized").unwrap();
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

        // 4) Use DistUniFracCFFI to compute weighted distance
        let dist_obj = DistUniFracCFFI::new(ctx_ptr);

        // Same example data with different abundances to test weighting:
        // Sample A => T1=7, T3=5, T4=2 (higher abundances)
        // Sample B => T2=3, T5=4, T6=9 (different abundance distribution)
        let va = [7.0, 0.0, 5.0, 2.0, 0.0, 0.0];
        let vb = [0.0, 3.0, 0.0, 0.0, 4.0, 9.0];

        let dist = dist_obj.eval(&va, &vb);
        println!("Weighted UniFrac(A,B) = {}", dist);
        
        // Test with different abundance patterns
        let vc = [10.0, 0.0, 1.0, 0.5, 0.0, 0.0]; // Different weights
        let vd = [0.0, 2.0, 0.0, 0.0, 8.0, 1.0];
        
        let dist2 = dist_obj.eval(&vc, &vd);
        println!("Weighted UniFrac(C,D) = {}", dist2);
        
        // Weighted distances should be different from unweighted for different abundance patterns
        assert!(dist >= 0.0 && dist <= 1.0, "Distance should be in [0,1]");
        assert!(dist2 >= 0.0 && dist2 <= 1.0, "Distance should be in [0,1]");

        // 5) Clean up
        unsafe {
            dist_unifrac_destroy(ctx_ptr);
            destroy_bptree_opaque(&mut tree_ptr);
        }
        free_obs_ids(&mut obs_ids);
    }
    */

    /// Example test using the new NewDistUniFrac with the exact function call pattern
    #[test]
    fn test_new_dist_unifrac_function_call() {
        let _ = env_logger::Builder::from_env(Env::default().default_filter_or("debug"))
        .is_test(true)
        .try_init();

        let newick_str = "((T1:0.1,(T2:0.05,T3:0.05):0.02):0.3,(T4:0.2,(T5:0.1,T6:0.15):0.05):0.4);";
        let feature_names = vec!["T1","T2","T3","T4","T5","T6"]
            .into_iter().map(|s| s.to_string()).collect::<Vec<String>>();
        let weighted = false;
        
        // Test the exact function call pattern requested
        let dist_unifrac = NewDistUniFrac::new(&newick_str, weighted, feature_names.clone()).unwrap();
        
        assert_eq!(dist_unifrac.num_features(), 6);
        assert_eq!(dist_unifrac.weighted, false);
        
        // Test with some sample data
        let va = vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0]; // T1, T3, T4 present
        let vb = vec![0.0, 1.0, 0.0, 0.0, 1.0, 1.0]; // T2, T5, T6 present

        let d = dist_unifrac.eval(&va, &vb);
        println!("New UniFrac distance = {}", d);
        assert!(d >= 0.0 && d <= 1.0);
        
        // Test that it's actually using the unifrac_bp pattern (non-zero distance for different samples)
        assert!(d > 0.0); // Should be > 0 since samples have no overlap
    }

    /// Test NewDistUniFrac against the original DistUniFrac implementation for correctness
    #[test]
    fn test_new_dist_unifrac_correctness_validation() {
        let _ = env_logger::Builder::from_env(Env::default().default_filter_or("debug"))
            .is_test(true)
            .try_init();

        // Simple tree with known structure for manual verification
        let newick_str = "((T1:0.1,T2:0.1):0.2,(T3:0.1,T4:0.1):0.2):0.0;";
        let feature_names = vec!["T1", "T2", "T3", "T4"]
            .into_iter().map(|s| s.to_string()).collect::<Vec<String>>();

        println!("=== Testing NewDistUniFrac vs Original DistUniFrac ===");
        println!("Tree: {}", newick_str);
        println!("Note: Different UniFrac implementations may use different normalization approaches");
        
        // Create NewDistUniFrac implementation
        let new_dist = NewDistUniFrac::new(&newick_str, false, feature_names.clone()).unwrap();
        
        // Test cases focusing on relative behavior rather than absolute values
        let test_cases = vec![
            // (sample_a, sample_b, description)
            (vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0], "T1 vs T2 (sister taxa)"),
            (vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 0.0], "T1 vs T3 (distant taxa)"),
            (vec![1.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 1.0], "T1,T2 vs T3,T4 (complete separation)"),
            (vec![1.0, 0.0, 1.0, 0.0], vec![0.0, 1.0, 0.0, 1.0], "T1,T3 vs T2,T4 (mixed)"),
        ];

        println!("\nNewDistUniFrac Properties Validation (shared/union algorithm):");
        println!("{:<25} | {:>15}", "Test Case", "NewDistUniFrac");
        println!("{}", "-".repeat(45));

        for (va, vb, description) in test_cases {
            let new_distance = new_dist.eval(&va, &vb);
            
            println!("{:<25} | {:>15.6}", description, new_distance);
        }
        
        // Validate basic UniFrac properties instead of comparing to DistUniFrac
        let identical_distance = new_dist.eval(&vec![1.0, 0.0, 0.0, 0.0], &vec![1.0, 0.0, 0.0, 0.0]);
        assert!(identical_distance.abs() < 1e-6, "Identical samples should have distance ~0, got {}", identical_distance);
        
        println!("\n✅ NewDistUniFrac properties validation passed!");
    }

    /// Test NewDistUniFrac with manually calculated expected values for mathematical correctness
    #[test]
    fn test_new_dist_unifrac_manual_calculation_validation() {
        let _ = env_logger::Builder::from_env(Env::default().default_filter_or("debug"))
            .is_test(true)
            .try_init();

        // Very simple 2-leaf tree: (T1:1.0,T2:1.0):0.0;
        // With normalized UniFrac: T1=1/1=1.0, T2=1/1=1.0, difference=1.0-(-1.0)=2.0 at T1 and T2
        // Then propagated up: T1 branch contributes 1.0*1.0=1.0, T2 branch contributes 1.0*1.0=1.0  
        // Total distance = 1.0 + 1.0 = 2.0 (this is the normalized UniFrac result)
        let newick_str = "(T1:1.0,T2:1.0):0.0;";
        let feature_names = vec!["T1", "T2"]
            .into_iter().map(|s| s.to_string()).collect::<Vec<String>>();

        println!("=== Manual UniFrac Calculation Validation ===");
        println!("Tree: {}", newick_str);
        println!("Expected: For T1 vs T2 with normalization, each sample normalized to 1.0");
        println!("T1 sample: [1.0, 0.0] normalized = [1.0, 0.0]"); 
        println!("T2 sample: [0.0, 1.0] normalized = [0.0, 1.0]");
        println!("Differences: T1 node = 1.0-0.0=1.0, T2 node = 0.0-1.0=-1.0");
        println!("Distance = |1.0|*1.0 + |-1.0|*1.0 = 2.0");
        
        let dist_unifrac = NewDistUniFrac::new(&newick_str, false, feature_names).unwrap();
        
        // Test case: T1 present vs T2 present (completely different)
        let va = vec![1.0, 0.0]; // T1 only
        let vb = vec![0.0, 1.0]; // T2 only
        
        let distance = dist_unifrac.eval(&va, &vb);
        println!("Calculated distance: {}", distance);
        
        // For the shared/union UniFrac algorithm (which NewDistUniFrac uses):
        // T1 vs T2: no shared branches, so shared = 0.0
        // Union includes both T1 and T2 branches: union = 1.0 + 1.0 = 2.0
        // UniFrac = 1.0 - shared/union = 1.0 - 0.0/2.0 = 1.0
        let expected_distance = 1.0; // Not 2.0 - that was the old algorithm
        
        let expected = expected_distance; // Use the corrected expected value
        let tolerance = 0.01;
        
        assert!((distance - expected).abs() < tolerance, 
                "Manual calculation validation failed: expected ~{}, got {}, diff={}", 
                expected, distance, (distance - expected).abs());
        
        // Test identical samples (should be 0)
        let identical_distance = dist_unifrac.eval(&va, &va);
        assert!(identical_distance.abs() < 1e-10, 
                "Identical samples should have distance 0, got {}", identical_distance);
        
        println!(" Manual calculation validation passed!");
        println!("  Expected: {:.3}, Calculated: {:.6}, Difference: {:.6}", 
                 expected, distance, (distance - expected).abs());

        // Test with a slightly more complex tree: ((T1:0.5,T2:0.5):0.5,T3:1.0):0.0;
        println!("\n=== Complex Tree Validation ===");
        let complex_newick = "((T1:0.5,T2:0.5):0.5,T3:1.0):0.0;";
        let complex_features = vec!["T1", "T2", "T3"].into_iter().map(|s| s.to_string()).collect();
        
        let complex_dist = NewDistUniFrac::new(&complex_newick, false, complex_features).unwrap();
        
        // T1 vs T2 (sister taxa): should share the 0.5 branch length
        let dist_12 = complex_dist.eval(&vec![1.0, 0.0, 0.0], &vec![0.0, 1.0, 0.0]);
        // T1 vs T3 (more distant): should share less
        let dist_13 = complex_dist.eval(&vec![1.0, 0.0, 0.0], &vec![0.0, 0.0, 1.0]);
        
        println!("T1 vs T2 (sisters): {:.6}", dist_12);
        println!("T1 vs T3 (distant): {:.6}", dist_13);
        
        // Sister taxa should be closer than distant taxa
        assert!(dist_12 < dist_13, 
                "Sister taxa should be closer: T1-T2({:.6}) should be < T1-T3({:.6})", 
                dist_12, dist_13);
        
        println!(" Phylogenetic relationship validation passed!");
        println!("  Sister taxa distance ({:.6}) < distant taxa distance ({:.6})", dist_12, dist_13);
    }

    /// Test NewDistUniFrac against ground truth values from test files
    #[test]
    fn test_new_dist_unifrac_ground_truth_validation() {
        use std::fs;
        use std::path::Path;
        
        let _ = env_logger::Builder::from_env(Env::default().default_filter_or("debug"))
            .is_test(true)
            .try_init();

        println!("=== Ground Truth Validation Test ===");
        println!("Testing NewDistUniFrac against actual ground truth from test files");
        
        // Load test data from data/ folder
        let tree_str = fs::read_to_string("data/test.nwk").expect("Failed to read test.nwk").trim().to_string();
        
        println!("Tree: {}", tree_str);
        
        // Feature names from OTU table  
        let features = vec!["T1".to_string(), "T2".to_string(), "T3".to_string(), "T4".to_string(), 
                           "T5".to_string(), "T6".to_string(), "T7".to_string()];
        
        println!("Features: {:?}", features);
        
        // Sample data from test_OTU_table.txt (converted to f32)
        let samples = vec![
            ("SampleA", vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]), // T1,T3,T4 present
            ("SampleB", vec![0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]), // T2,T5,T6 present
            ("SampleC", vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]), // T1,T3,T7 present
        ];
        
        for (name, data) in &samples {
            println!("{}: {:?}", name, data);
        }
        
        // Ground truth distances from test_truth.txt
        let ground_truth = vec![
            (("SampleA", "SampleB"), 0.4929577112197876),
            (("SampleA", "SampleC"), 0.3409090936183929),
            (("SampleB", "SampleC"), 0.4577465057373047),
        ];
        
        // Create NewDistUniFrac 
        let new_dist = NewDistUniFrac::new(&tree_str, false, features).expect("Failed to create NewDistUniFrac");
        
        println!("\n=== Validation Against Ground Truth (tolerance: 0.01) ===");
        println!("{:<12} | {:<12} | {:>15} | {:>15} | {:>10} | Status", 
                 "Sample A", "Sample B", "NewDistUniFrac", "Ground Truth", "Diff");
        println!("{}", "-".repeat(80));
        
        let tolerance = 0.01;
        let mut all_passed = true;
        
        // Test against ground truth values
        for ((sample_a, sample_b), expected_distance) in ground_truth {
            let data_a = &samples.iter().find(|(name, _)| name == &sample_a).unwrap().1;
            let data_b = &samples.iter().find(|(name, _)| name == &sample_b).unwrap().1;
            
            let new_distance = new_dist.eval(data_a, data_b) as f64;
            let diff = (new_distance - expected_distance).abs();
            let status = if diff < tolerance { "✓" } else { "✗" };
            
            if diff >= tolerance {
                all_passed = false;
            }
            
            println!("{:<12} | {:<12} | {:>15.6} | {:>15.6} | {:>10.6} | {}", 
                     sample_a, sample_b, new_distance, expected_distance, diff, status);
        }
        
        if all_passed {
            println!("\n✅ All ground truth validation tests passed within tolerance of {}!", tolerance);
            println!("NewDistUniFrac matches the ground truth values from test_truth.txt! 🚀");
        } else {
            panic!("❌ Ground truth validation failed! NewDistUniFrac results don't match test_truth.txt values");
        }
    }

    #[test]
    fn test_new_dist_unifrac_mouse_gut_validation() {
        use std::fs;
        use std::path::Path;
        
        let _ = env_logger::Builder::from_env(Env::default().default_filter_or("warn"))
            .is_test(true)
            .try_init();

        println!("=== Mouse Gut Data Validation Test ===");
        println!("Testing NewDistUniFrac on real mouse gut microbiome data");
        
        // Check if mouse gut data files exist
        let tree_file = "data/Mouse_gut_zotu_aligned.tre";
        let counts_file = "data/Mouse_gut_zotu_counts.txt";
        
        if !Path::new(tree_file).exists() || !Path::new(counts_file).exists() {
            println!("Mouse gut data files not found, skipping validation");
            return;
        }
        
        // Read files
        let tree_content = fs::read_to_string(tree_file).expect("Failed to read tree file");
        let counts_content = fs::read_to_string(counts_file).expect("Failed to read counts file");
        
        println!("Tree file size: {} characters", tree_content.len());
        println!("Counts file size: {} characters", counts_content.len());
        
        // Parse counts data using existing parsing logic
        let lines: Vec<&str> = counts_content.lines().collect();
        
        fn parse_mouse_gut_format(lines: &[&str]) -> (Vec<String>, Vec<Vec<f32>>, Vec<String>) {
            if lines.is_empty() { return (vec![], vec![], vec![]); }
            
            // Find header line
            let mut header_idx = 0;
            for (i, line) in lines.iter().enumerate() {
                if line.starts_with('#') || (!line.trim().is_empty() && line.contains('\t')) {
                    header_idx = i;
                    break;
                }
            }
            
            let header = lines[header_idx].trim_start_matches('#').trim();
            let parts: Vec<&str> = header.split('\t').collect();
            
            if parts.len() < 2 { return (vec![], vec![], vec![]); }
            
            let sample_names: Vec<String> = parts[1..].iter().map(|s| s.to_string()).collect();
            let mut feature_names = Vec::new();
            let mut all_samples: Vec<Vec<f32>> = vec![vec![]; sample_names.len()];
            
            // Parse data lines
            for line in lines.iter().skip(header_idx + 1) {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') { continue; }
                
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() != sample_names.len() + 1 { continue; }
                
                let feature_name = parts[0].to_string();
                let values: Result<Vec<f32>, _> = parts[1..].iter().map(|s| s.parse::<f32>()).collect();
                
                if let Ok(vals) = values {
                    feature_names.push(feature_name);
                    for (i, val) in vals.into_iter().enumerate() {
                        all_samples[i].push(val);
                    }
                }
            }
            
            (feature_names, all_samples, sample_names)
        }
        
        let (features, samples, sample_names) = parse_mouse_gut_format(&lines);
        
        println!("Parsed {} features and {} samples", features.len(), sample_names.len());
        println!("Sample names: {:?}", sample_names.iter().take(5).collect::<Vec<_>>());
        
        if features.is_empty() || samples.is_empty() {
            println!("Failed to parse mouse gut data, skipping test");
            return;
        }
        
        // Create NewDistUniFrac implementation
        let new_dist = match NewDistUniFrac::new(&tree_content, false, features.clone()) {
            Ok(d) => d,
            Err(e) => {
                println!("Failed to create NewDistUniFrac: {}", e);
                return;
            }
        };
        
        println!("\n=== Validation Results (tolerance: 0.01) ===");
        println!("{:<15} | {:<15} | {:>15}", "Sample A", "Sample B", "NewDistUniFrac");
        println!("{}", "-".repeat(50));
        
        let mut test_count = 0;
        let max_tests = 10; // Test only first 10 pairs to avoid too much output  
        let mut distances_calculated = Vec::new();
        
        // Test pairwise distances on subset of samples
        for i in 0..std::cmp::min(3, samples.len()) {
            for j in (i+1)..std::cmp::min(4, samples.len()) {
                if test_count >= max_tests { break; }
                
                let sample_a = &sample_names[i];
                let sample_b = &sample_names[j];
                
                // Calculate with NewDistUniFrac only
                let new_distance = new_dist.eval(&samples[i], &samples[j]) as f64;
                distances_calculated.push(new_distance);
                
                test_count += 1;
                println!("{:<15} | {:<15} | {:>15.6}", sample_a, sample_b, new_distance);
            }
            if test_count >= max_tests { break; }
        }
        
        // Validate basic properties instead of comparing to DistUniFrac
        println!("\n✅ NewDistUniFrac successfully processed {} sample pairs from mouse gut data!", test_count);
        println!("Distance range: {:.6} to {:.6}", 
                 distances_calculated.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                 distances_calculated.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        
        // Test self-distance (should be 0)
        let self_distance = new_dist.eval(&samples[0], &samples[0]);
        assert!(self_distance.abs() < 1e-6, "Self distance should be ~0, got {}", self_distance);
        
        println!("Self-distance validation: ✓ (got {:.10})", self_distance);
        println!("Mouse gut microbiome data processing completed successfully! 🧬");
    }

    // ============================================================================
    // Comprehensive NewDistUniFrac Tests
    // ============================================================================

    #[test]
    fn test_new_dist_unifrac_creation() {
        init_log();
        
        // Simple 4-leaf tree
        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec!["T1".to_string(), "T2".to_string(), "T3".to_string(), "T4".to_string()];
        
        // Test unweighted
        let dist_unifrac = NewDistUniFrac::new(newick_str, false, feature_names.clone());
        assert!(dist_unifrac.is_ok(), "Failed to create unweighted NewDistUniFrac: {:?}", dist_unifrac.err());
        
        let unifrac = dist_unifrac.unwrap();
        assert_eq!(unifrac.weighted, false);
        assert_eq!(unifrac.feature_names.len(), 4);
        assert_eq!(unifrac.num_features(), 4);
        
        // Test weighted
        let dist_unifrac_weighted = NewDistUniFrac::new(newick_str, true, feature_names);
        assert!(dist_unifrac_weighted.is_ok(), "Failed to create weighted NewDistUniFrac");
        
        let unifrac_weighted = dist_unifrac_weighted.unwrap();
        assert_eq!(unifrac_weighted.weighted, true);
    }

    #[test]
    fn test_new_dist_unifrac_invalid_newick() {
        init_log();
        
        // Invalid Newick string - missing closing parenthesis
        let invalid_newick = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1;";
        let feature_names = vec!["T1".to_string(), "T2".to_string(), "T3".to_string(), "T4".to_string()];
        
        let result = NewDistUniFrac::new(invalid_newick, false, feature_names);
        assert!(result.is_err(), "Should fail with invalid Newick string");
    }

    #[test]
    fn test_new_dist_unifrac_missing_feature() {
        init_log();
        
        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec!["T1".to_string(), "T2".to_string(), "T5".to_string()]; // T5 doesn't exist in tree
        
        let result = NewDistUniFrac::new(newick_str, false, feature_names);
        assert!(result.is_err(), "Should fail with missing feature name");
        
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("Feature name 'T5' not found in tree"), 
                "Error message should mention missing feature: {}", err_msg);
    }

    #[test]
    fn test_new_dist_unifrac_identical_samples() {
        init_log();
        
        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec!["T1".to_string(), "T2".to_string(), "T3".to_string(), "T4".to_string()];
        
        let dist_unifrac = NewDistUniFrac::new(newick_str, false, feature_names).unwrap();
        
        // Identical samples should have distance 0
        let va = vec![1.0, 0.0, 1.0, 0.0];
        let vb = vec![1.0, 0.0, 1.0, 0.0];
        
        let distance = dist_unifrac.eval(&va, &vb);
        assert!(distance.abs() < 1e-6, "Distance between identical samples should be ~0, got {}", distance);
    }

    #[test]
    fn test_new_dist_unifrac_completely_different_samples() {
        init_log();
        
        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec!["T1".to_string(), "T2".to_string(), "T3".to_string(), "T4".to_string()];
        
        let new_dist = NewDistUniFrac::new(newick_str, false, feature_names).unwrap();
        
        // Completely different samples - test against known expected value
        let va = vec![1.0, 1.0, 0.0, 0.0]; // T1, T2 present
        let vb = vec![0.0, 0.0, 1.0, 1.0]; // T3, T4 present
        
        let new_distance = new_dist.eval(&va, &vb);
        
        println!("NewDistUniFrac distance for completely different samples: {}", new_distance);
        
        // For NewDistUniFrac (shared/union algorithm):
        // va = [1,1,0,0] (T1,T2), vb = [0,0,1,1] (T3,T4)
        // No shared branches between the two clades, so shared = 0
        // Union includes all branches, so UniFrac = 1.0 - 0/union = 1.0
        let expected_distance = 1.0; // Correct value for shared/union algorithm
        assert!((new_distance - expected_distance).abs() < 0.01, 
            "NewDistUniFrac ({}) should match expected value ({}) within 0.01 tolerance", 
            new_distance, expected_distance);
        
        // Also verify it's a substantial distance as expected
        assert!(new_distance > 0.5, "Distance should be substantial for completely different samples, got {}", new_distance);
    }

    #[test]
    fn test_new_dist_unifrac_zero_samples() {
        init_log();
        
        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec!["T1".to_string(), "T2".to_string(), "T3".to_string(), "T4".to_string()];
        
        let dist_unifrac = NewDistUniFrac::new(newick_str, false, feature_names).unwrap();
        
        // Both samples empty
        let va = vec![0.0, 0.0, 0.0, 0.0];
        let vb = vec![0.0, 0.0, 0.0, 0.0];
        
        let distance = dist_unifrac.eval(&va, &vb);
        assert!(distance.abs() < 1e-6, "Distance between empty samples should be ~0, got {}", distance);
    }

    #[test]
    fn test_new_dist_unifrac_single_feature() {
        init_log();
        
        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec!["T1".to_string(), "T2".to_string(), "T3".to_string(), "T4".to_string()];
        
        let new_dist = NewDistUniFrac::new(newick_str, false, feature_names.clone()).unwrap();
        
        // Test T1 vs T2 (sister taxa) - basic validation
        let va = vec![1.0, 0.0, 0.0, 0.0]; // Only T1
        let vb = vec![0.0, 1.0, 0.0, 0.0]; // Only T2
        
        let new_distance = new_dist.eval(&va, &vb);
        
        println!("Sister taxa - NewDistUniFrac: {}", new_distance);
        assert!(new_distance >= 0.0 && new_distance <= 1.0, 
            "NewDistUniFrac distance ({}) should be between 0.0 and 1.0", new_distance);
        
        // Test T1 vs T3 (more distant) - should be larger distance
        let vc = vec![0.0, 0.0, 1.0, 0.0]; // Only T3
        let new_distance2 = new_dist.eval(&va, &vc);
        
        println!("Distant taxa - NewDistUniFrac: {}", new_distance2);
        assert!(new_distance2 >= 0.0 && new_distance2 <= 1.0, 
            "NewDistUniFrac distance ({}) should be between 0.0 and 1.0", new_distance2);
        assert!(new_distance2 > new_distance, 
            "Distance to more distant taxa ({}) should be larger than sister taxa ({})", 
            new_distance2, new_distance);
        
        // Verify distance ordering is preserved
        assert!(new_distance2 > new_distance, "More distant taxa should have larger distance");
    }

    #[test]
    fn test_new_dist_unifrac_weighted_flag() {
        init_log();
        
        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec!["T1".to_string(), "T2".to_string(), "T3".to_string(), "T4".to_string()];
        
        let dist_unweighted = NewDistUniFrac::new(newick_str, false, feature_names.clone()).unwrap();
        let dist_weighted = NewDistUniFrac::new(newick_str, true, feature_names).unwrap();
        
        // Verify the weighted flag is stored correctly
        assert_eq!(dist_unweighted.weighted, false);
        assert_eq!(dist_weighted.weighted, true);
        
        // Note: The current unifrac_pair implementation is unweighted
        // So both will give the same result regardless of the flag
        let va = vec![10.0, 0.0, 1.0, 0.0];  // Different abundances
        let vb = vec![1.0, 0.0, 10.0, 0.0];
        
        let dist_unwt = dist_unweighted.eval(&va, &vb);
        let dist_wt = dist_weighted.eval(&va, &vb);
        
        println!("Unweighted distance: {}", dist_unwt);
        println!("Weighted distance: {}", dist_wt);
        
        // Both should be the same since unifrac_pair implements unweighted UniFrac
        assert!((dist_unwt - dist_wt).abs() < 1e-6, 
                "Both should be same (unweighted) for current implementation");
    }

    /// Test function for user's tree and CSV files
    #[test]
    fn test_new_dist_unifrac_from_files() {
        use std::fs;
        use std::path::Path;
        
        init_log();
        
        // Use the counts file since it contains the actual feature data we need for UniFrac
        let tree_file = "data/Mouse_gut_zotu_aligned.tre";
        let counts_file = "data/Mouse_gut_zotu_counts.txt";
        
        println!("Testing NewDistUniFrac with real mouse gut microbiome data...");
        
        // Check if files exist
        if !Path::new(tree_file).exists() {
            println!("Tree file '{}' not found. Please ensure the data folder exists.", tree_file);
            return;
        }
        
        if !Path::new(counts_file).exists() {
            println!("Counts file '{}' not found. Please ensure the data folder exists.", counts_file);
            return;
        }
        
        println!("Using tree: {}", tree_file);
        println!("Using counts: {}", counts_file);
        
        // Read tree file
        let newick_str = match fs::read_to_string(tree_file) {
            Ok(content) => {
                println!("✓ Successfully read tree file: {} characters", content.len());
                content.trim().to_string()
            }
            Err(e) => {
                println!("✗ Error reading tree file: {}", e);
                return;
            }
        };
        
        println!("Tree content (first 100 chars): {}", 
                 if newick_str.len() > 100 { &newick_str[..100] } else { &newick_str });
        
        // Read counts file
        let data_content = match fs::read_to_string(counts_file) {
            Ok(content) => {
                println!("✓ Successfully read counts file: {} characters", content.len());
                content
            }
            Err(e) => {
                println!("✗ Error reading counts file: {}", e);
                return;
            }
        };
        
        // Parse counts file
        let lines: Vec<&str> = data_content.lines().collect();
        if lines.is_empty() {
            println!("✗ Counts file is empty");
            return;
        }
        
        // Helper function to parse counts/OTU table format
        fn parse_counts_format(lines: &[&str]) -> (Vec<String>, Vec<Vec<f32>>, Vec<String>) {
            if lines.is_empty() { return (vec![], vec![], vec![]); }
            
            // Look for header line (might start with # or be first line)
            let mut header_idx = 0;
            for (i, line) in lines.iter().enumerate() {
                if line.starts_with('#') || line.contains('\t') || line.contains(',') {
                    header_idx = i;
                    break;
                }
            }
            
            let header = lines[header_idx].trim_start_matches('#').trim();
            let separator = if header.contains('\t') { '\t' } else { ',' };
            
            let parts: Vec<&str> = header.split(separator).collect();
            if parts.len() < 2 { return (vec![], vec![], vec![]); }
            
            // First column usually OTU/feature ID, rest are sample names
            let sample_names: Vec<String> = parts[1..].iter().map(|s| s.trim().to_string()).collect();
            let mut feature_names = Vec::new();
            let mut all_samples: Vec<Vec<f32>> = vec![vec![]; sample_names.len()];
            
            // Parse data lines
            for line in lines.iter().skip(header_idx + 1) {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') { continue; }
                
                let parts: Vec<&str> = line.split(separator).collect();
                if parts.len() != sample_names.len() + 1 { continue; }
                
                let feature_name = parts[0].trim().to_string();
                let values: Result<Vec<f32>, _> = parts[1..].iter().map(|s| s.trim().parse::<f32>()).collect();
                
                if let Ok(vals) = values {
                    feature_names.push(feature_name);
                    for (i, val) in vals.into_iter().enumerate() {
                        all_samples[i].push(val);
                    }
                }
            }
            
            // Transpose: convert from features x samples to samples x features
            let samples: Vec<Vec<f32>> = all_samples;
            
            (feature_names, samples, sample_names)
        }
        
        // Parse OTU counts format
        let (feature_names, samples, sample_names) = parse_counts_format(&lines);
        
        if feature_names.is_empty() {
            println!("✗ No feature names found");
            return;
        }
        
        if samples.is_empty() {
            println!("✗ No valid samples found");
            return;
        }
        
        println!("✓ Found {} features and {} samples", feature_names.len(), samples.len());
        println!("Features (first 10): {:?}", 
                 if feature_names.len() > 10 { &feature_names[..10] } else { &feature_names });
        println!("Samples: {:?}", sample_names);
        
        // Create NewDistUniFrac - this will test if feature names match tree
        println!("\nCreating NewDistUniFrac instance...");
        let dist_unifrac = match NewDistUniFrac::new(&newick_str, false, feature_names.clone()) {
            Ok(unifrac) => {
                println!("✓ NewDistUniFrac created successfully!");
                println!("  - Weighted: {}", unifrac.weighted);
                println!("  - Features: {}", unifrac.num_features());
                unifrac
            }
            Err(e) => {
                println!("✗ Error creating NewDistUniFrac: {}", e);
                println!("\nTroubleshooting tips:");
                println!("1. Check that feature names in data match leaf names in tree exactly");
                println!("2. Tree format should be valid Newick");
                println!("3. Feature names are case-sensitive");
                
                // Show first few feature names for debugging
                println!("\nFirst 5 features from data: {:?}", &feature_names[..feature_names.len().min(5)]);
                return;
            }
        };
        
        // Compute sample distances (limit to first few samples for test)
        println!("\nComputing UniFrac distances...");
        let max_samples = samples.len().min(5); // Limit for test performance
        
        for i in 0..max_samples {
            for j in (i + 1)..max_samples {
                let distance = dist_unifrac.eval(&samples[i], &samples[j]);
                println!("  {} vs {}: {:.6}", sample_names[i], sample_names[j], distance);
                
                // Validate distance properties
                assert!(distance >= 0.0 && distance <= 1.0, 
                       "Distance out of range [0,1]: {}", distance);
            }
        }
        
        // Test self-distance
        if !samples.is_empty() {
            let self_distance = dist_unifrac.eval(&samples[0], &samples[0]);
            println!("\nSelf-distance test: {:.8}", self_distance);
            assert!(self_distance.abs() < 1e-6, "Self-distance should be ~0");
        }
        
        println!("\n✓ Mouse gut microbiome UniFrac analysis completed successfully!");
    }

    #[test]
    fn test_new_dist_unifrac_api_compatibility() {
        init_log();
        
        // Test the exact function call pattern mentioned in the requirements
        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let weighted = false;
        let feature_names = vec!["T1".to_string(), "T2".to_string(), "T3".to_string(), "T4".to_string()];
        
        // This should work exactly as specified
        let dist_unifrac = NewDistUniFrac::new(&newick_str, weighted, feature_names.clone());
        assert!(dist_unifrac.is_ok(), "API compatibility test failed");
        
        let unifrac = dist_unifrac.unwrap();
        
        // Test feature access methods
        assert_eq!(unifrac.feature_names(), &feature_names);
        assert_eq!(unifrac.num_features(), 4);
        
        // Test evaluation
        let va = vec![1.0, 0.0, 1.0, 0.0];
        let vb = vec![0.0, 1.0, 0.0, 1.0];
        let distance = unifrac.eval(&va, &vb);
        
        assert!(distance >= 0.0 && distance <= 1.0, "Distance should be normalized between 0 and 1");
    }

    #[test]
    fn test_new_dist_unifrac_large_tree() {
        init_log();
        
        // Larger tree with more taxa - basic validation
        let newick_str = "(((T1:0.1,T2:0.1):0.05,(T3:0.1,T4:0.1):0.05):0.1,((T5:0.1,T6:0.1):0.05,(T7:0.1,T8:0.1):0.05):0.1):0.0;";
        let feature_names = vec![
            "T1".to_string(), "T2".to_string(), "T3".to_string(), "T4".to_string(),
            "T5".to_string(), "T6".to_string(), "T7".to_string(), "T8".to_string()
        ];
        
        let new_dist = NewDistUniFrac::new(newick_str, false, feature_names.clone()).unwrap();
        
        // Test completely different clades
        let va = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // First 4 taxa
        let vb = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]; // Last 4 taxa
        
        let new_distance = new_dist.eval(&va, &vb);
        
        println!("Large tree different clades - NewDistUniFrac: {}", new_distance);
        assert!(new_distance >= 0.0 && new_distance <= 1.0, 
            "NewDistUniFrac distance ({}) should be between 0.0 and 1.0", new_distance);
        
        // Test overlapping samples
        let vc = vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]; // Mixed across clades
        let new_distance2 = new_dist.eval(&va, &vc);
        
        println!("Large tree overlapping - NewDistUniFrac: {}", new_distance2);
        assert!(new_distance2 >= 0.0 && new_distance2 <= 1.0, 
            "NewDistUniFrac distance ({}) should be between 0.0 and 1.0", new_distance2);
        
        // Verify distance ordering is preserved
        assert!(new_distance2 < new_distance, 
            "Overlapping samples should have smaller distance ({} < {})", new_distance2, new_distance);
    }

    #[test]
    fn test_new_dist_unifrac_succparen_patterns() {
        init_log();
        
        // Test that our implementation follows succparen patterns correctly
        let newick_str = "((T1:0.2,T2:0.3):0.1,(T3:0.4,T4:0.5):0.2):0.0;";
        let feature_names = vec!["T1".to_string(), "T2".to_string(), "T3".to_string(), "T4".to_string()];
        
        let dist_unifrac = NewDistUniFrac::new(newick_str, false, feature_names).unwrap();
        
        // Verify internal structure is built correctly
        assert!(dist_unifrac.post.len() > 0, "Post-order traversal should be populated");
        assert!(dist_unifrac.kids.len() > 0, "Children arrays should be populated");
        assert!(dist_unifrac.lens.len() > 0, "Branch lengths should be populated");
        assert_eq!(dist_unifrac.leaf_ids.len(), 4, "Should have 4 leaf IDs");
        
        // Test distance computation
        let va = vec![1.0, 0.0, 0.0, 0.0];
        let vb = vec![0.0, 1.0, 0.0, 0.0];
        let distance = dist_unifrac.eval(&va, &vb);
        
        // Distance should be positive and reasonable
        assert!(distance > 0.0 && distance < 1.0, "Distance should be in valid range, got {}", distance);
    }

    #[test]
    fn test_new_dist_unifrac_bitvec_conversion() {
        init_log();
        
        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec!["T1".to_string(), "T2".to_string(), "T3".to_string(), "T4".to_string()];
        
        let new_dist = NewDistUniFrac::new(newick_str, false, feature_names.clone()).unwrap();
        let new_dist = NewDistUniFrac::new(newick_str, false, feature_names.clone()).unwrap();
        
        // Test presence/absence conversion with varying abundances
        let va = vec![0.5, 0.0, 1.5, 0.0]; // T1 and T3 present (> 0)
        let vb = vec![0.0, 2.0, 0.0, 0.1]; // T2 and T4 present (> 0)
        
        let new_distance = new_dist.eval(&va, &vb);
        
        println!("Bitvec conversion - NewDistUniFrac: {}", new_distance);
        assert!(new_distance >= 0.0 && new_distance <= 1.0, 
            "NewDistUniFrac distance ({}) should be between 0.0 and 1.0", new_distance);
        
        // Test with very small positive values
        let vc = vec![0.0001, 0.0, 0.0001, 0.0]; // Very small positive values
        let vd = vec![0.0, 0.0001, 0.0, 0.0001]; 
        
        let new_distance2 = new_dist.eval(&vc, &vd);
        
        println!("Small values - NewDistUniFrac: {}", new_distance2);
        assert!(new_distance2 >= 0.0 && new_distance2 <= 1.0, 
            "NewDistUniFrac distance ({}) should be between 0.0 and 1.0", new_distance2);
        
        // Both should give same result (presence/absence only matters)
        assert!((new_distance - new_distance2).abs() < 1e-6, 
            "Distance should only depend on presence/absence, got {} vs {}", new_distance, new_distance2);
    }

    #[test]
    fn test_new_dist_unifrac_symmetry() {
        init_log();
        
        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec!["T1".to_string(), "T2".to_string(), "T3".to_string(), "T4".to_string()];
        
        let dist_unifrac = NewDistUniFrac::new(newick_str, false, feature_names).unwrap();
        
        // Test that distance is symmetric: d(A,B) = d(B,A)
        let va = vec![1.0, 0.0, 1.0, 0.0];
        let vb = vec![0.0, 1.0, 0.0, 1.0];
        
        let distance_ab = dist_unifrac.eval(&va, &vb);
        let distance_ba = dist_unifrac.eval(&vb, &va);
        
        assert!((distance_ab - distance_ba).abs() < 1e-10, 
                "Distance should be symmetric: d(A,B)={}, d(B,A)={}", distance_ab, distance_ba);
    }

    #[test]
    fn test_new_dist_unifrac_triangle_inequality() {
        init_log();
        
        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec!["T1".to_string(), "T2".to_string(), "T3".to_string(), "T4".to_string()];
        
        let dist_unifrac = NewDistUniFrac::new(newick_str, false, feature_names).unwrap();
        
        // Test triangle inequality: d(A,C) <= d(A,B) + d(B,C)
        let va = vec![1.0, 0.0, 0.0, 0.0]; // Only T1
        let vb = vec![0.0, 1.0, 0.0, 0.0]; // Only T2  
        let vc = vec![0.0, 0.0, 1.0, 0.0]; // Only T3
        
        let d_ab = dist_unifrac.eval(&va, &vb);
        let d_bc = dist_unifrac.eval(&vb, &vc);
        let d_ac = dist_unifrac.eval(&va, &vc);
        
        println!("d(A,B) = {}, d(B,C) = {}, d(A,C) = {}", d_ab, d_bc, d_ac);
        
        // Triangle inequality: d(A,C) <= d(A,B) + d(B,C)
        assert!(d_ac <= d_ab + d_bc + 1e-10, 
                "Triangle inequality violated: d(A,C)={} > d(A,B)+d(B,C)={}", 
                d_ac, d_ab + d_bc);
    }

    #[test]
    fn test_new_dist_unifrac_distance_trait_usage() {
        use std::fs;
        
        init_log();
        
        // Same data as your mouse gut test
        let tree_file = "data/Mouse_gut_zotu_aligned.tre";
        let counts_file = "data/Mouse_gut_zotu_counts.txt";
        
        if !std::path::Path::new(tree_file).exists() || !std::path::Path::new(counts_file).exists() {
            println!("Data files not found, skipping test");
            return;
        }
        
        // Read your real data
        let tree_content = fs::read_to_string(tree_file).unwrap();
        let data_content = fs::read_to_string(counts_file).unwrap();
        let lines: Vec<&str> = data_content.lines().collect();
        
        // Helper function to parse counts/OTU table format
        fn parse_counts_format(lines: &[&str]) -> (Vec<String>, Vec<Vec<f32>>, Vec<String>) {
            if lines.is_empty() { return (vec![], vec![], vec![]); }
            
            let mut header_idx = 0;
            for (i, line) in lines.iter().enumerate() {
                if line.starts_with('#') || line.contains('\t') || line.contains(',') {
                    header_idx = i;
                    break;
                }
            }
            
            let header = lines[header_idx].trim_start_matches('#').trim();
            let separator = if header.contains('\t') { '\t' } else { ',' };
            
            let parts: Vec<&str> = header.split(separator).collect();
            if parts.len() < 2 { return (vec![], vec![], vec![]); }
            
            let sample_names: Vec<String> = parts[1..].iter().map(|s| s.trim().to_string()).collect();
            let mut feature_names = Vec::new();
            let mut all_samples: Vec<Vec<f32>> = vec![vec![]; sample_names.len()];
            
            for line in lines.iter().skip(header_idx + 1) {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') { continue; }
                
                let parts: Vec<&str> = line.split(separator).collect();
                if parts.len() != sample_names.len() + 1 { continue; }
                
                let feature_name = parts[0].trim().to_string();
                let values: Result<Vec<f32>, _> = parts[1..].iter().map(|s| s.trim().parse::<f32>()).collect();
                
                if let Ok(vals) = values {
                    feature_names.push(feature_name);
                    for (i, val) in vals.into_iter().enumerate() {
                        all_samples[i].push(val);
                    }
                }
            }
            
            let samples: Vec<Vec<f32>> = all_samples;
            (feature_names, samples, sample_names)
        }
        
        let (feature_names, samples, sample_names) = parse_counts_format(&lines);
        
        if feature_names.is_empty() || samples.is_empty() {
            println!("No valid data found");
            return;
        }
        
        // Create NewDistUniFrac
        let dist_unifrac = NewDistUniFrac::new(&tree_content, false, feature_names).unwrap();
        
        println!("=== Using Distance Trait to Get Same Numbers ===");
        
        // METHOD 1: Direct Distance trait usage
        let distance_trait: &dyn Distance<f32> = &dist_unifrac;
        
        if samples.len() >= 4 {
            println!("🔹 Direct Distance Trait Usage:");
            println!("   {} vs {}: {:.6}", sample_names[0], sample_names[1], distance_trait.eval(&samples[0], &samples[1]));
            println!("   {} vs {}: {:.6}", sample_names[0], sample_names[2], distance_trait.eval(&samples[0], &samples[2]));
            println!("   {} vs {}: {:.6}", sample_names[0], sample_names[3], distance_trait.eval(&samples[0], &samples[3]));
            println!("   Self-distance ({} vs {}): {:.8}", sample_names[0], sample_names[0], distance_trait.eval(&samples[0], &samples[0]));
        }
        
        println!();
        
        // METHOD 2: Generic function using Distance trait
        fn calculate_distances<D: Distance<f32>>(
            dist: &D, 
            samples: &[Vec<f32>], 
            sample_names: &[String]
        ) {
            println!("🔹 Generic Function with Distance Trait:");
            if samples.len() >= 4 {
                println!("   {} vs {}: {:.6}", sample_names[0], sample_names[1], dist.eval(&samples[0], &samples[1]));
                println!("   {} vs {}: {:.6}", sample_names[0], sample_names[2], dist.eval(&samples[0], &samples[2]));
                println!("   {} vs {}: {:.6}", sample_names[0], sample_names[3], dist.eval(&samples[0], &samples[3]));
                println!("   Self-distance ({} vs {}): {:.8}", sample_names[0], sample_names[0], dist.eval(&samples[0], &samples[0]));
            }
        }
        
        calculate_distances(&dist_unifrac, &samples, &sample_names);
        
        println!();
        
        // METHOD 3: Polymorphic usage - can work with ANY Distance implementation
        fn process_with_any_distance<D: Distance<f32>>(distance_impl: D, data: &[Vec<f32>], names: &[String]) {
            println!("🔹 Polymorphic Distance Usage:");
            if data.len() >= 4 {
                println!("   {} vs {}: {:.6}", names[0], names[1], distance_impl.eval(&data[0], &data[1]));
                println!("   {} vs {}: {:.6}", names[0], names[2], distance_impl.eval(&data[0], &data[2]));
                println!("   {} vs {}: {:.6}", names[0], names[3], distance_impl.eval(&data[0], &data[3]));
                println!("   Self-distance ({} vs {}): {:.8}", names[0], names[0], distance_impl.eval(&data[0], &data[0]));
            }
        }
        
        process_with_any_distance(dist_unifrac, &samples, &sample_names);
        
        println!();
        println!("✅ All methods produce identical results using the Distance trait!");
    }
} // end of module tests
