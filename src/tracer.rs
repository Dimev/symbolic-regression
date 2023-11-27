use std::simd::f32x64;

use rayon::prelude::*;

struct Trace {
    /// emissive,how much light is reflected
    emissive: f32x64,

    /// opacity, how much light is absorbed
    opacity: f32x64,
}

pub struct TraceResult {
    /// emissive,how much light is reflected
    pub emissive: Vec<f32>,

    /// opacity, how much light is absorbed
    pub opacity: Vec<f32>,
}

pub struct Params {
    /// start height
    pub height: Vec<f32>,

    /// start azimuth (dot up)
    pub dot_up: Vec<f32>,

    /// scale height
    pub scale: Vec<f32>,

    /// optical depth
    pub density: Vec<f32>,

    /// light direction
    pub dot_l: Vec<f32>,
}

impl Params {
    pub fn new(n: usize) -> Self {
        Self {
            height: (0..n).into_par_iter().map(|_| fastrand::f32()).collect(),
            dot_up: (0..n).into_par_iter().map(|_| fastrand::f32()).collect(),
            scale: (0..n).into_par_iter().map(|_| fastrand::f32()).collect(),
            density: (0..n).into_par_iter().map(|_| fastrand::f32()).collect(),
            dot_l: (0..n).into_par_iter().map(|_| fastrand::f32()).collect(),
        }
    }

    pub fn trace(&self) -> TraceResult {
        let combo: Vec<Trace> = self
            .height
            .par_chunks_exact(64)
            .zip(self.dot_up.par_chunks_exact(64))
            .zip(self.scale.par_chunks_exact(64))
            .zip(self.density.par_chunks_exact(64))
            .zip(self.dot_l.par_chunks_exact(64))
            .map(|((((a, b), c), d), e)| (a, b, c, d, e))
            .map(|(a, b, c, d, e)| {
                (
                    f32x64::from_slice(a),
                    f32x64::from_slice(b),
                    f32x64::from_slice(c),
                    f32x64::from_slice(d),
                    f32x64::from_slice(e),
                )
            })
            .map(|(a, b, c, d, e)| trace(a, b, c, d, e))
            .collect();

        TraceResult {
            emissive: combo
                .par_iter()
                .flat_map_iter(|x| x.emissive.to_array())
                .collect(),
            opacity: combo
                .par_iter()
                .flat_map_iter(|x| x.opacity.to_array())
                .collect(),
        }
    }
}

fn trace(height: f32x64, dot_up: f32x64, scale: f32x64, density: f32x64, dot_l: f32x64) -> Trace {
    Trace {
        emissive: f32x64::splat(0.0),
        opacity: f32x64::splat(0.0),
    }
}
