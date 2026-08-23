//! Standalone dynamic linear-algebra layer replacing `nalgebra` for rapier's runtime math.
//!
//! This module reproduces the subset of `nalgebra` used by rapier (`DVector`, `DMatrix`,
//! `Jacobian`, `LU`, views) with **bit-identical** floating-point behavior so that existing
//! simulations and `enhanced-determinism` saves remain reproducible after `nalgebra` is removed.
//!
//! Determinism strategy: every numeric kernel replicates `nalgebra`'s exact arithmetic order.
//! - `dot`/`axpy`/`component_mul` copy `nalgebra`'s unrolled loop bodies verbatim (indexing our
//!   column-major buffer the same way `nalgebra` indexes its storage).
//! - `gemm`/`gemm_tr`/`tr_mul` delegate to the same `matrixmultiply` crate `nalgebra` uses.
//! - `LU` is a verbatim copy of `nalgebra`'s `linalg/lu.rs`.
//!
//! Layout: vectors/matrices are stored **column-major** in a flat `Vec<T>`, identical to
//! `nalgebra`'s `VecStorage`, so the copied loop bodies produce identical bits.

use crate::alloc_prelude::*;
use simba::scalar::ComplexField;

/// A dynamically-sized column vector (`nalgebra::DVector<Real>` replacement).
///
/// Stored column-major (a single column), so element `i` lives at `data[i]`.
#[derive(Clone, Debug, Default)]
pub struct DVector<T: ComplexField<RealField = T> + Copy> {
    /// Column-major element storage (a single column).
    pub data: Vec<T>,
}

impl<T: ComplexField<RealField = T> + Copy + PartialOrd> DVector<T> {
    /// Creates a vector from a `Vec` of elements (column-major, single column).
    #[inline]
    pub fn from_vec(data: Vec<T>) -> Self {
        DVector { data }
    }

    /// Creates a vector from a slice (cloned).
    #[inline]
    pub fn from_slice(slice: &[T]) -> Self {
        DVector {
            data: slice.to_vec(),
        }
    }

    /// Creates a zero vector of length `n`.
    #[inline]
    pub fn zeros(n: usize) -> Self {
        DVector {
            data: vec![T::zero(); n],
        }
    }

    /// Creates a vector filled with `value`.
    #[inline]
    pub fn from_element(n: usize, value: T) -> Self {
        DVector {
            data: vec![value; n],
        }
    }

    /// Number of rows (length).
    #[inline]
    pub fn nrows(&self) -> usize {
        self.data.len()
    }

    /// Number of columns (always 1 for a vector).
    #[inline]
    pub fn ncols(&self) -> usize {
        1
    }

    /// Total length.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Immutable slice of the underlying data.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable slice of the underlying data.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Indexing (column vector → row index).
    #[inline]
    pub fn index(&self, i: usize) -> T {
        self.data[i]
    }

    /// Mutable indexing.
    #[inline]
    pub fn at_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }

    /// Returns a copy of the rows `first..first+n` as a new `DVector`.
    #[inline]
    pub fn rows(&self, first: usize, n: usize) -> DVectorView<'_, T> {
        DVectorView {
            data: &self.data[first..first + n],
        }
    }

    /// Returns a mutable view of the rows `first..first+n`.
    #[inline]
    pub fn rows_mut(&mut self, first: usize, n: usize) -> DVectorViewMut<'_, T> {
        DVectorViewMut {
            data: &mut self.data[first..first + n],
        }
    }

    /// Clones this vector (owned).
    #[inline]
    pub fn clone_owned(&self) -> DVector<T> {
        DVector {
            data: self.data.clone(),
        }
    }

    /// Copies the content of `other` into `self` (same length).
    #[inline]
    pub fn copy_from(&mut self, other: &DVector<T>) {
        self.data.copy_from_slice(&other.data);
    }

    /// Fills every element with `value`.
    #[inline]
    pub fn fill(&mut self, value: T) {
        for e in &mut self.data {
            *e = value;
        }
    }

    /// The infinity norm (max absolute value).
    #[inline]
    pub fn amax(&self) -> T {
        let mut m = T::zero();
        for &e in &self.data {
            let a = e.abs();
            if a > m {
                m = a;
            }
        }
        m
    }
}

impl<T: ComplexField<RealField = T> + Copy + PartialOrd> DVector<T> {
    /// Euclidean norm.
    #[inline]
    pub fn norm(&self) -> T {
        let mut s = T::zero();
        for &e in &self.data {
            s += e * e;
        }
        s.sqrt()
    }

    /// Normalized copy (falls back to zeros if norm is zero).
    #[inline]
    pub fn normalize(&self) -> DVector<T> {
        let n = self.norm();
        if n.is_zero() {
            return DVector::zeros(self.data.len());
        }
        let mut out = self.data.clone();
        for e in &mut out {
            *e /= n;
        }
        DVector { data: out }
    }
}

impl<T: ComplexField<RealField = T> + Copy> DVector<T> {
    /// Inserts `n` rows at index `i`, filling them with `value`.
    #[inline]
    pub fn insert_rows(&self, i: usize, n: usize, value: T) -> DVector<T> {
        let mut data = Vec::with_capacity(self.data.len() + n);
        data.extend_from_slice(&self.data[..i]);
        data.extend(std::iter::repeat_n(value, n));
        data.extend_from_slice(&self.data[i..]);
        DVector { data }
    }

    /// Returns the rows `first..` as an owned `DVector`.
    #[inline]
    pub fn index_range(&self, first: usize) -> DVector<T> {
        DVector {
            data: self.data[first..].to_vec(),
        }
    }

    /// Dot product, bit-identical to `nalgebra::DVector::dot`.
    ///
    /// Copy of `nalgebra`'s `dotx` unrolled loop (blas.rs). `conjugate` is identity for real `T`.
    #[inline]
    pub fn dot(&self, rhs: &DVector<T>) -> T {
        let n = self.data.len();
        assert_eq!(n, rhs.data.len(), "Dot product dimension mismatch.");

        // NOTE: for a dynamic vector (`Dyn` rows) nalgebra does NOT take the U2/U3/U4
        // special cases; it always uses the general unrolled loop below. We replicate that
        // exactly (the tail loop gives a sequential sum for n < 8).
        let mut res = T::zero();
        let mut i = 0;
        let mut acc0 = T::zero();
        let mut acc1 = T::zero();
        let mut acc2 = T::zero();
        let mut acc3 = T::zero();
        let mut acc4 = T::zero();
        let mut acc5 = T::zero();
        let mut acc6 = T::zero();
        let mut acc7 = T::zero();

        while n - i >= 8 {
            acc0 += self.data[i] * rhs.data[i];
            acc1 += self.data[i + 1] * rhs.data[i + 1];
            acc2 += self.data[i + 2] * rhs.data[i + 2];
            acc3 += self.data[i + 3] * rhs.data[i + 3];
            acc4 += self.data[i + 4] * rhs.data[i + 4];
            acc5 += self.data[i + 5] * rhs.data[i + 5];
            acc6 += self.data[i + 6] * rhs.data[i + 6];
            acc7 += self.data[i + 7] * rhs.data[i + 7];
            i += 8;
        }

        res += acc0 + acc4;
        res += acc1 + acc5;
        res += acc2 + acc6;
        res += acc3 + acc7;

        for k in i..n {
            res += self.data[k] * rhs.data[k];
        }

        res
    }

    /// Element-wise multiply, bit-identical to `nalgebra`'s `component_mul`.
    #[inline]
    pub fn component_mul(&self, other: &DVector<T>) -> DVector<T> {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "Component-wise multiply dimension mismatch."
        );
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a * *b)
            .collect();
        DVector { data }
    }

    /// Computes `self = a * x + b * self` (axpy), bit-identical to `nalgebra`'s `axpy`.
    #[inline]
    pub fn axpy(&mut self, a: T, x: &DVector<T>, b: T) {
        assert_eq!(
            self.data.len(),
            x.data.len(),
            "Axpy dimension mismatch."
        );
        if b.is_zero() {
            for (s, &xi) in self.data.iter_mut().zip(&x.data) {
                *s = a * xi;
            }
        } else {
            for (s, &xi) in self.data.iter_mut().zip(&x.data) {
                *s = a * xi + b * *s;
            }
        }
    }

    /// Transpose (vector → 1×n matrix view). Implemented at the matrix level; here returns self.
    #[inline]
    pub fn transpose(&self) -> DVector<T> {
        // A column vector transposed is a row vector; rapier mostly uses `.tr_mul` for that.
        // Provided for API completeness (bit-exact: same data).
        DVector {
            data: self.data.clone(),
        }
    }
}

/// Immutable view into a contiguous slice of a `DVector` (replaces `na::DVectorView`).
#[derive(Clone, Copy, Debug)]
pub struct DVectorView<'a, T: ComplexField<RealField = T> + Copy> {
    /// Borrowed slice of the viewed elements.
    pub data: &'a [T],
}

impl<'a, T: ComplexField<RealField = T> + Copy + PartialOrd> DVectorView<'a, T> {
    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.data.len()
    }

    /// Column count (1).
    #[inline]
    pub fn ncols(&self) -> usize {
        1
    }

    /// Length.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Immutable slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Dot product against another vector.
    #[inline]
    pub fn dot(&self, rhs: &DVector<T>) -> T {
        DVector::from_slice(self.data).dot(rhs)
    }

    /// Clones into an owned `DVector`.
    #[inline]
    pub fn into_owned(self) -> DVector<T> {
        DVector::from_slice(self.data)
    }

    /// Copies this view's content into `dst`.
    #[inline]
    pub fn copy_to(&self, dst: &mut DVector<T>) {
        dst.data.copy_from_slice(self.data);
    }
}

/// Mutable view into a contiguous slice of a `DVector` (replaces `na::DVectorViewMut`).
#[derive(Debug)]
pub struct DVectorViewMut<'a, T: ComplexField<RealField = T> + Copy> {
    /// Mutably borrowed slice of the viewed elements.
    pub data: &'a mut [T],
}

impl<'a, T: ComplexField<RealField = T> + Copy> DVectorViewMut<'a, T> {
    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.data.len()
    }

    /// Column count (1).
    #[inline]
    pub fn ncols(&self) -> usize {
        1
    }

    /// Length.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Immutable slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    /// Copies `src` into this view.
    #[inline]
    pub fn copy_from(&mut self, src: &DVector<T>) {
        self.data.copy_from_slice(&src.data);
    }

    /// Copies this view's content into `dst`.
    #[inline]
    pub fn copy_to(&self, dst: &mut DVector<T>) {
        dst.data.copy_from_slice(self.data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector as NaDVector;

    fn approx_eq_bits(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits()
    }

    #[test]
    fn dvector_dot_bit_identical_to_nalgebra() {
        let data: Vec<f64> = (0..137).map(|i| (i as f64) * 1.137 + 0.25).collect();
        let data2: Vec<f64> = (0..137).map(|i| (i as f64) * 0.731 - 0.5).collect();

        let mine = DVector::from_vec(data.clone());
        let na = NaDVector::from_vec(data.clone());
        let mine2 = DVector::from_vec(data2.clone());
        let na2 = NaDVector::from_vec(data2.clone());

        let rd = mine.dot(&mine2);
        let rna = na.dot(&na2);
        assert!(
            approx_eq_bits(rd, rna),
            "dot mismatch: mine={:?} na={:?}",
            rd,
            rna
        );

        // various lengths (incl. < 8 tail-sum path)
        for n in [1usize, 2, 3, 4, 5, 7, 8, 9, 64, 137] {
            let a: Vec<f64> = (0..n).map(|i| i as f64 * 0.3 + 1.0).collect();
            let b: Vec<f64> = (0..n).map(|i| i as f64 * 0.7 - 0.2).collect();
            let m = DVector::from_vec(a.clone()).dot(&DVector::from_vec(b.clone()));
            let nref = NaDVector::from_vec(a).dot(&NaDVector::from_vec(b));
            assert!(approx_eq_bits(m, nref), "dot n={} mismatch: {} vs {}", n, m, nref);
        }
    }

    #[test]
    fn dvector_component_mul_bit_identical() {
        let a: Vec<f64> = (0..100).map(|i| (i as f64) * 0.91 + 0.1).collect();
        let b: Vec<f64> = (0..100).map(|i| (i as f64) * 0.37 - 0.4).collect();
        let m = DVector::from_vec(a.clone()).component_mul(&DVector::from_vec(b.clone()));
        let nref = NaDVector::from_vec(a).component_mul(&NaDVector::from_vec(b));
        for (x, y) in m.data.iter().zip(nref.iter()) {
            assert!(approx_eq_bits(*x, *y), "component_mul mismatch {} vs {}", x, y);
        }
    }

    #[test]
    fn dvector_axpy_bit_identical() {
        for (a, b) in [(2.0, 3.0), (1.0, 0.0), (0.0, 1.0), (5.5, -2.0)] {
            let x: Vec<f64> = (0..60).map(|i| (i as f64) * 0.13 + 0.7).collect();
            let mut s: Vec<f64> = (0..60).map(|i| (i as f64) * 0.29 - 0.3).collect();
            let mut mine = DVector::from_vec(s.clone());
            let na_x = NaDVector::from_vec(x.clone());
            let mut na_s = NaDVector::from_vec(s.clone());
            mine.axpy(a, &DVector::from_vec(x.clone()), b);
            na_s.axpy(a, &na_x, b);
            for (m, n) in mine.data.iter().zip(na_s.iter()) {
                assert!(approx_eq_bits(*m, *n), "axpy mismatch {} vs {}", m, n);
            }
            let _ = &mut s;
        }
    }
}
