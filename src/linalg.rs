//! Standalone dynamic linear-algebra layer replacing `nalgebra` for rapier's runtime math.
//!
//! This module reproduces the subset of `nalgebra` used by rapier (`DVector`, `DMatrix`,
//! `Jacobian`, `LU`, views) with **bit-identical** floating-point behavior so that existing
//! simulations and `enhanced-determinism` saves remain reproducible after `nalgebra` is removed.
//!
//! Determinism strategy: every numeric kernel replicates `nalgebra`'s exact arithmetic order.
//! - `dot`/`axpy`/`component_mul` copy `nalgebra`'s unrolled loop bodies verbatim (indexing our
//!   column-major buffer the same way `nalgebra` indexes its storage).
//! - `gemm`/`gemm_tr`/`tr_mul` delegate to the same `matrixmultiply` crate `nalgebra` uses
//!   (only when at least one dimension is dynamic and all relevant dims > 5, matching
//!   `nalgebra`'s `gemm_uninit` dispatch).
//! - `LU` (P4) is a verbatim copy of `nalgebra`'s `linalg/lu.rs`.
//!
//! Layout: vectors/matrices are stored **column-major** in a flat `Vec<T>`, identical to
//! `nalgebra`'s `VecStorage`, so the copied loop bodies produce identical bits.
//!
//! Stride model: every matrix knows its `(nrows, ncols, col_stride)`. For a full matrix
//! `col_stride == nrows`. For a row-block view (`fixed_rows` / `rows_range`) the column stride
//! is **inherited from the parent** (a row block keeps the parent's inter-column distance), so
//! element `(i, j)` of a view with base offset `off` is at `data[off + i + j * col_stride]`.
//! This exactly mirrors `nalgebra`'s `RawStorage` strides and is what makes bit-identical `gemm`
//! over mixed-row matrices (e.g. a 3-row view × 6-row matrix × 3-row view) work.

use crate::alloc_prelude::*;
use simba::scalar::ComplexField;
use std::ops::Deref;
use std::ops::DerefMut;

/// A dynamically-sized column vector (`nalgebra::DVector<Real>` replacement).
///
/// Stored column-major (a single column), so element `i` lives at `data[i]`.
#[derive(Clone, Debug, Default)]
pub struct DVector<T: ComplexField<RealField = T> + Copy> {
    /// Column-major element storage (a single column).
    pub data: Vec<T>,
}

impl<T: ComplexField<RealField = T> + Copy> DVector<T> {
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
        self.data.fill(value);
    }

    /// Computes `self = alpha * a * x + beta * self`, bit-identical to `nalgebra`'s vector `gemv`.
    /// `a` is a column-readable matrix (`ColAccess`), `x` a vector. Generic over matrix kind.
    #[inline]
    pub fn gemv<M: ColAccess<T>>(&mut self, alpha: T, a: &M, x: &DVectorView<T>, beta: T) {
        let n = a.ncols();
        assert_eq!(a.nrows(), self.data.len(), "gemv: row mismatch");
        assert_eq!(n, x.data.len(), "gemv: col mismatch");
        for j in 0..n {
            let a_col = a.col(j);
            let c = x.data[j];
            // y = alpha * a_col * c + (j==0 ? beta : 1) * y   (axcpy semantics)
            let b = if j == 0 { beta } else { T::one() };
            axpy_col(&mut self.data, alpha, &a_col.data, c, b);
        }
    }

    /// Borrows this vector as an immutable view.
    #[inline]
    pub fn as_view(&self) -> DVectorView<'_, T> {
        DVectorView { data: &self.data }
    }

    /// Transpose (vector → 1×n matrix view). Implemented at the matrix level; here returns self.
    #[inline]
    pub fn transpose(&self) -> DVector<T> {
        DVector {
            data: self.data.clone(),
        }
    }

    /// Builds a vector of length `n` where element `i` is `f(i)`.
    #[inline]
    pub fn from_fn(n: usize, mut f: impl FnMut(usize) -> T) -> DVector<T> {
        let data = (0..n).map(&mut f).collect();
        DVector { data }
    }

    /// Returns an owned copy of rows `first..first+n`.
    #[inline]
    pub fn rows_owned(&self, first: usize, n: usize) -> DVector<T> {
        DVector {
            data: self.data[first..first + n].to_vec(),
        }
    }
}

impl<T: ComplexField<RealField = T> + Copy> From<Vec<T>> for DVector<T> {
    #[inline]
    fn from(v: Vec<T>) -> DVector<T> {
        DVector { data: v }
    }
}

impl<T: ComplexField<RealField = T> + Copy> From<&[T]> for DVector<T> {
    #[inline]
    fn from(v: &[T]) -> DVector<T> {
        DVector {
            data: v.to_vec(),
        }
    }
}

impl<T: ComplexField<RealField = T> + Copy + PartialOrd> DVector<T> {
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
        assert_eq!(self.data.len(), x.data.len(), "Axpy dimension mismatch.");
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

    /// Computes `self = alpha * a.transpose() * x + beta * self`, bit-identical to
    /// `nalgebra`'s vector `gemv_tr`. `a` is a column-readable matrix (`ColAccess`).
    #[inline]
    pub fn gemv_tr<M: ColAccess<T>>(&mut self, alpha: T, a: &M, x: &DVectorView<T>, beta: T) {
        let mut v = DVectorViewMut {
            data: self.as_mut_slice(),
        };
        v.gemv_tr(alpha, a, x, beta);
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

impl<T: ComplexField<RealField = T> + Copy> std::ops::Index<usize> for DVectorView<'_, T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.data[i]
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

    /// Computes `self = a * x + b * self` (axpy), bit-identical to `nalgebra`'s `axpy`.
    #[inline]
    pub fn axpy(&mut self, a: T, x: &DVector<T>, b: T) {
        assert_eq!(self.data.len(), x.data.len(), "Axpy dimension mismatch.");
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

    /// Computes `self = alpha * a.transpose() * x + beta * self`, bit-identical to
    /// `nalgebra`'s vector `gemv_tr`. `a` is a column-readable matrix (`ColAccess`).
    #[inline]
    pub fn gemv_tr<M: ColAccess<T>>(&mut self, alpha: T, a: &M, x: &DVectorView<T>, beta: T) {
        let n = a.ncols();
        assert_eq!(a.nrows(), x.data.len(), "gemv_tr: row mismatch");
        assert_eq!(n, self.data.len(), "gemv_tr: col mismatch");
        for j in 0..n {
            let d = a.col(j).dot(&DVector::from_slice(x.data));
            if !beta.is_zero() {
                self.data[j] = alpha * d + beta * self.data[j];
            } else {
                self.data[j] = alpha * d;
            }
        }
    }

    /// Euclidean norm.
    #[inline]
    pub fn norm(&self) -> T {
        let mut s = T::zero();
        for &e in self.data.iter() {
            s += e * e;
        }
        s.sqrt()
    }

    /// Squared Euclidean norm.
    #[inline]
    pub fn norm_squared(&self) -> T {
        let mut s = T::zero();
        for &e in self.data.iter() {
            s += e * e;
        }
        s
    }

    /// Fills every element with `value`.
    #[inline]
    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }

    /// Returns a sub-view of rows `r0..` as a mutable vector view.
    #[inline]
    pub fn rows_range_mut(&mut self, r0: usize) -> DVectorViewMut<'_, T> {
        DVectorViewMut {
            data: &mut self.data[r0..],
        }
    }

    /// Element-wise multiply-assign `self[i] *= other[i]`.
    #[inline]
    pub fn component_mul_assign(&mut self, other: &DVector<T>) {
        assert_eq!(self.data.len(), other.data.len());
        for i in 0..self.data.len() {
            self.data[i] *= other.data[i];
        }
    }
}

impl<T: ComplexField<RealField = T> + Copy> std::ops::Index<usize> for DVectorViewMut<'_, T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

impl<T: ComplexField<RealField = T> + Copy> std::ops::IndexMut<usize> for DVectorViewMut<'_, T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}

/// A dynamically-sized matrix stored **column-major** in a flat `Vec<T>` with an explicit
/// column stride.
///
/// Indexing: element `(i, j)` (row `i`, column `j`) lives at `data[i + j * col_stride]`.
/// For a full matrix `col_stride == nrows`, matching `nalgebra`'s `VecStorage` layout. For a
/// row-block view the column stride is inherited from the parent so numeric kernels produce
/// identical bits to `nalgebra`.
#[derive(Clone, Debug, Default)]
pub struct DMatrix<T: ComplexField<RealField = T> + Copy> {
    /// Column-major element storage.
    pub data: Vec<T>,
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Distance (in elements) between element `(i, j)` and `(i, j+1)`.
    pub col_stride: usize,
}

impl<T: ComplexField<RealField = T> + Copy> DMatrix<T> {
    /// Creates a zero `nrows × ncols` matrix (column stride = `nrows`).
    #[inline]
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        DMatrix {
            data: vec![T::zero(); nrows * ncols],
            nrows,
            ncols,
            col_stride: nrows,
        }
    }

    /// Creates a matrix from row-major data (`nalgebra`'s `from_row_slice` semantics).
    #[inline]
    pub fn from_row_slice(nrows: usize, ncols: usize, slice: &[T]) -> Self {
        let mut data = vec![T::zero(); nrows * ncols];
        for i in 0..nrows {
            for j in 0..ncols {
                data[i + j * nrows] = slice[i * ncols + j];
            }
        }
        DMatrix {
            data,
            nrows,
            ncols,
            col_stride: nrows,
        }
    }

    /// Column-major raw pointer (for `matrixmultiply` delegation).
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Column-major raw mutable pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Raw strides `(row_stride, col_stride)` for `matrixmultiply` delegation.
    #[inline]
    pub fn strides(&self) -> (usize, usize) {
        (1, self.col_stride)
    }

    /// Returns a copy of column `j`.
    #[inline]
    pub fn column(&self, j: usize) -> DVector<T> {
        let start = j * self.col_stride;
        DVector {
            data: self.data[start..start + self.nrows].to_vec(),
        }
    }

    /// Returns a mutable view of column `j`.
    #[inline]
    pub fn column_mut(&mut self, j: usize) -> DVectorViewMut<'_, T> {
        let start = j * self.col_stride;
        DVectorViewMut {
            data: &mut self.data[start..start + self.nrows],
        }
    }

    /// Fills every element with `value`.
    #[inline]
    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }

    /// Copies the content of `other` (same dimensions) into `self`.
    #[inline]
    pub fn copy_from(&mut self, other: &DMatrix<T>) {
        assert_eq!(
            (self.nrows, self.ncols),
            (other.nrows, other.ncols),
            "copy_from dimension mismatch"
        );
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                let dst = i + j * self.col_stride;
                let src = i + j * other.col_stride;
                self.data[dst] = other.data[src];
            }
        }
    }

    /// Returns an owned clone.
    #[inline]
    pub fn clone_owned(&self) -> DMatrix<T> {
        DMatrix {
            data: self.data.clone(),
            nrows: self.nrows,
            ncols: self.ncols,
            col_stride: self.col_stride,
        }
    }

    /// Transpose (bit-identical to `nalgebra::Matrix::transpose`).
    #[inline]
    pub fn transpose(&self) -> DMatrix<T> {
        let mut data = vec![T::zero(); self.nrows * self.ncols];
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                data[j + i * self.ncols] = self.data[i + j * self.col_stride];
            }
        }
        DMatrix {
            data,
            nrows: self.ncols,
            ncols: self.nrows,
            col_stride: self.ncols,
        }
    }

    /// Immutable element access, matching nalgebra's `matrix[(i, j)]`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> T {
        self.data[i + j * self.col_stride]
    }

    /// Mutable element access.
    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.data[i + j * self.col_stride]
    }

    /// Swaps elements `(i0, j0)` and `(i1, j1)` in place.
    #[inline]
    pub fn swap(&mut self, (i0, j0): (usize, usize), (i1, j1): (usize, usize)) {
        self.data
            .swap(i0 + j0 * self.col_stride, i1 + j1 * self.col_stride);
    }

    /// Whether the matrix is square.
    #[inline]
    pub fn is_square(&self) -> bool {
        self.nrows == self.ncols
    }

    /// Returns a mutable view of columns `first..first+n`.
    #[inline]
    pub fn columns_mut(&mut self, first: usize, n: usize) -> MatrixViewMut<'_, T> {
        MatrixViewMut {
            data: &mut self.data[first * self.col_stride..(first + n) * self.col_stride],
            nrows: self.nrows,
            ncols: n,
            col_stride: self.col_stride,
        }
    }

    /// Copies `src` (length `nrows`) into column `j`.
    #[inline]
    pub fn set_column(&mut self, j: usize, src: &DVector<T>) {
        let start = j * self.col_stride;
        self.data[start..start + self.nrows].copy_from_slice(&src.data);
    }

    /// Returns an immutable view of columns `first..first+n`.
    #[inline]
    pub fn columns(&self, first: usize, n: usize) -> MatrixView<'_, T> {
        MatrixView {
            data: &self.data[first * self.col_stride..(first + n) * self.col_stride],
            nrows: self.nrows,
            ncols: n,
            col_stride: self.col_stride,
        }
    }

    /// Returns the `L` rows starting at row `first` (shifting the column-major data by `first`).
    /// Matches nalgebra's `fixed_rows::<L>(first)`.
    #[inline]
    pub fn fixed_rows<const L: usize>(&self, first: usize) -> MatrixView<'_, T> {
        let ncols = self.ncols;
        MatrixView {
            data: &self.data[first..first + L + (ncols - 1) * self.col_stride],
            nrows: L,
            ncols,
            col_stride: self.col_stride,
        }
    }

    /// Mutable `L` rows block starting at `first` (column stride inherited from parent).
    #[inline]
    pub fn fixed_rows_mut<const L: usize>(&mut self, first: usize) -> MatrixViewMut<'_, T> {
        let ncols = self.ncols;
        MatrixViewMut {
            data: &mut self.data[first..first + L + (ncols - 1) * self.col_stride],
            nrows: L,
            ncols,
            col_stride: self.col_stride,
        }
    }

    /// Returns the rows `r0..r0+len` as a new matrix view (column stride inherited).
    /// Matches nalgebra's `rows_range(r0..r0+len)`.
    #[inline]
    pub fn rows_range(&self, r0: usize, len: usize) -> MatrixView<'_, T> {
        let ncols = self.ncols;
        MatrixView {
            data: &self.data[r0..r0 + len + (ncols - 1) * self.col_stride],
            nrows: len,
            ncols,
            col_stride: self.col_stride,
        }
    }

    /// Mutable `rows_range` view.
    #[inline]
    pub fn rows_range_mut(&mut self, r0: usize, len: usize) -> MatrixViewMut<'_, T> {
        let ncols = self.ncols;
        MatrixViewMut {
            data: &mut self.data[r0..r0 + len + (ncols - 1) * self.col_stride],
            nrows: len,
            ncols,
            col_stride: self.col_stride,
        }
    }

    /// Returns a pair of mutable row-range views `(rows 0..n0, rows n0..n0+n1)`.
    /// Matches nalgebra's `rows_range_pair_mut(0..n0, n0..n0+n1)`. The two views partition the
    /// row axis (disjoint element sets) while sharing the parent's column stride.
    #[inline]
    pub fn rows_range_pair_mut(
        &mut self,
        n0: usize,
        n1: usize,
    ) -> (MatrixViewMut<'_, T>, MatrixViewMut<'_, T>) {
        let ncols = self.ncols;
        let cs = self.col_stride;
        let left_end = n0 + (ncols - 1) * cs;
        let right = &mut self.data[n0..];
        let (left, right) = right.split_at_mut(left_end - n0);
        (
            MatrixViewMut {
                data: left,
                nrows: n0,
                ncols,
                col_stride: cs,
            },
            MatrixViewMut {
                data: &mut right[..n1 + (ncols - 1) * cs],
                nrows: n1,
                ncols,
                col_stride: cs,
            },
        )
    }

    /// Computes `self = alpha * a * b + beta * self`, bit-identical to `nalgebra`'s `gemm`.
    ///
    /// Replicates `nalgebra`'s exact dispatch (blas_uninit.rs):
    /// - large dynamic matrices → `matrixmultiply::dgemm` with column-major strides;
    /// - otherwise → per-column `gemv`/`axcpy` loop.
    #[inline]
    pub fn gemm<Ma: MatrixLike<T>, Mb: MatrixLike<T>>(
        &mut self,
        alpha: T,
        a: &Ma,
        b: &Mb,
        beta: T,
    ) {
        let nrows1 = self.nrows;
        let ncols1 = self.ncols;
        let nrows2 = a.nrows();
        let ncols2 = a.ncols();
        let nrows3 = b.nrows();
        let ncols3 = b.ncols();

        assert_eq!(
            ncols2, nrows3,
            "gemm: dimensions mismatch for multiplication."
        );
        assert_eq!(
            (nrows1, ncols1),
            (nrows2, ncols3),
            "gemm: dimensions mismatch for addition."
        );

        const SMALL_DIM: usize = 5;
        let large =
            nrows1 > SMALL_DIM && ncols1 > SMALL_DIM && nrows2 > SMALL_DIM && ncols2 > SMALL_DIM;

        if large {
            // matrixmultiply path (identical to nalgebra for f64). Strides come from the views.
            let (rsa, csa) = a.strides();
            let (rsb, csb) = b.strides();
            let (rsc, csc) = self.strides();
            unsafe {
                matrixmultiply::dgemm(
                    nrows2,
                    ncols2,
                    ncols3,
                    std::mem::transmute_copy(&alpha),
                    a.as_ptr() as *const f64,
                    rsa as isize,
                    csa as isize,
                    b.as_ptr() as *const f64,
                    rsb as isize,
                    csb as isize,
                    std::mem::transmute_copy(&beta),
                    self.as_mut_ptr() as *mut f64,
                    rsc as isize,
                    csc as isize,
                );
            }
            return;
        }

        // Small/static fallback: per-column gemv, matching nalgebra's gemv_uninit/axcpy.
        // NOTE: nalgebra passes the original `beta` to *every* column's gemv; the within-column
        // first-iteration special-casing (beta only for j==0 of the inner loop, `1` thereafter)
        // is handled inside gemv_column.
        for j1 in 0..ncols1 {
            let mut y = self.column_mut(j1);
            let bcol = b.col(j1);
            gemv_column(&mut y, alpha, a, &bcol.as_view(), beta);
        }
    }

    /// Rank-1 update `self += alpha * x * y^T + beta * self`, bit-identical to `nalgebra::ger`.
    /// `x` has `self.nrows` elements, `y` has `self.ncols` elements.
    #[inline]
    pub fn ger(&mut self, alpha: T, x: &DVector<T>, y: &DVector<T>, beta: T) {
        assert_eq!(self.nrows, x.len(), "ger: x dimension mismatch");
        assert_eq!(self.ncols, y.len(), "ger: y dimension mismatch");
        for j in 0..self.ncols {
            let val = alpha * y.data[j];
            let mut col = self.column_mut(j);
            col.axpy(val, x, beta);
        }
    }

    /// Computes `self = alpha * rhs.transpose() * mid * rhs + beta * self` (quadratic form),
    /// bit-identical to `nalgebra::quadform`.
    #[inline]
    pub fn quadform<M: MatrixLike<T>>(&mut self, alpha: T, mid: &M, rhs: &M, beta: T) {
        let dim = mid.nrows();
        assert_eq!(mid.ncols(), dim, "quadform: mid must be square");
        assert_eq!(rhs.nrows(), dim, "quadform: rhs rows must match mid");
        assert_eq!(
            self.nrows,
            rhs.ncols(),
            "quadform: self rows must match rhs cols"
        );
        assert_eq!(
            self.ncols,
            rhs.ncols(),
            "quadform: self cols must match rhs cols"
        );

        let mut work = DVector::zeros(dim);
        for j in 0..rhs.ncols() {
            work.gemv(T::one(), mid, &rhs.col(j).as_view(), T::zero());
            self.column_mut(j)
                .gemv_tr(alpha, rhs, &work.as_view(), if j == 0 { beta } else { T::one() });
        }
    }

    /// Equivalent to `self.transpose() * rhs` but stores the result (vector) into `out`.
    /// Bit-identical to `nalgebra::tr_mul_to` for the matrix × vector case.
    #[inline]
    pub fn tr_mul_to(&self, rhs: &DVectorView<T>, out: &mut DVector<T>) {
        assert_eq!(self.nrows, rhs.data.len(), "tr_mul_to: row mismatch");
        assert_eq!(self.ncols, out.len(), "tr_mul_to: out mismatch");
        out.gemv_tr(T::one(), self, rhs, T::zero());
    }

    /// Computes `self = alpha * a.transpose() * b + beta * self`, bit-identical to
    /// `nalgebra::gemm_tr`.
    #[inline]
    pub fn gemm_tr<Ma: MatrixLike<T>, Mb: MatrixLike<T>>(
        &mut self,
        alpha: T,
        a: &Ma,
        b: &Mb,
        beta: T,
    ) {
        let nrows1 = self.nrows;
        let ncols1 = self.ncols;
        let nrows2 = a.nrows();
        let ncols2 = a.ncols();
        let nrows3 = b.nrows();
        let ncols3 = b.ncols();

        assert_eq!(
            nrows2, nrows3,
            "gemm_tr: dimensions mismatch for multiplication."
        );
        assert_eq!(
            (nrows1, ncols1),
            (ncols2, ncols3),
            "gemm_tr: dimensions mismatch for addition."
        );

        for j1 in 0..ncols1 {
            let mut y = self.column_mut(j1);
            let bcol = b.col(j1);
            gemv_tr_column(&mut y, alpha, a, &bcol.as_view(), beta);
        }
    }

    /// Returns an immutable view of the block starting at `(first_row, first_col)` with
    /// dimensions `(nrows, ncols)`. Matches nalgebra's `matrix.view((r, c), (nr, nc))`.
    #[inline]
    pub fn view(&self, (r, c): (usize, usize), (nr, nc): (usize, usize)) -> MatrixView<'_, T> {
        MatrixView {
            data: &self.data[r + c * self.col_stride
                ..r + c * self.col_stride + nr + (nc - 1) * self.col_stride],
            nrows: nr,
            ncols: nc,
            col_stride: self.col_stride,
        }
    }

    /// Returns a mutable view of the block starting at `(first_row, first_col)` with
    /// dimensions `(nrows, ncols)`.
    #[inline]
    pub fn view_mut(
        &mut self,
        (r, c): (usize, usize),
        (nr, nc): (usize, usize),
    ) -> MatrixViewMut<'_, T> {
        MatrixViewMut {
            data: &mut self.data[r + c * self.col_stride
                ..r + c * self.col_stride + nr + (nc - 1) * self.col_stride],
            nrows: nr,
            ncols: nc,
            col_stride: self.col_stride,
        }
    }

    /// Returns a mutable view of rows `first..first+n` (as a row-block of this matrix).
    #[inline]
    pub fn rows_mut_block(&mut self, first: usize, n: usize) -> MatrixViewMut<'_, T> {
        MatrixViewMut {
            data: &mut self.data[first..first + n + (self.ncols - 1) * self.col_stride],
            nrows: n,
            ncols: self.ncols,
            col_stride: self.col_stride,
        }
    }

    /// Returns a mutable view of the columns in `range` (e.g. `start..` or `a..b`).
    #[inline]
    pub fn columns_range_mut(&mut self, range: std::ops::Range<usize>) -> MatrixViewMut<'_, T> {
        let n = range.end - range.start;
        MatrixViewMut {
            data: &mut self.data[range.start * self.col_stride
                ..(range.start + n) * self.col_stride],
            nrows: self.nrows,
            ncols: n,
            col_stride: self.col_stride,
        }
    }
}

impl<T: ComplexField<RealField = T> + Copy> std::ops::Index<(usize, usize)> for DMatrix<T> {
    type Output = T;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.data[i + j * self.col_stride]
    }
}

impl<T: ComplexField<RealField = T> + Copy> std::ops::IndexMut<(usize, usize)> for DMatrix<T> {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        &mut self.data[i + j * self.col_stride]
    }
}

/// A fixed-size `R × C` matrix, stored **column-major** (column stride = `R`).
///
/// This is the `nalgebra::SMatrix<Real, R, C>` replacement used by `multibody.rs` for the
/// `SPATIAL_DIM × SPATIAL_DIM` rigid-body mass matrix and `tmp` temporaries. It wraps a
/// `DMatrix<R, C>` so all matrix kernels (gemm, views, etc.) work uniformly; `Deref`/`DerefMut`
/// expose the inner `DMatrix` directly.
#[derive(Clone, Debug, Default)]
pub struct SMatrix<T: ComplexField<RealField = T> + Copy, const R: usize, const C: usize> {
    /// Inner `R × C` column-major matrix (column stride = `R`).
    pub matrix: DMatrix<T>,
}

impl<T: ComplexField<RealField = T> + Copy, const R: usize, const C: usize> SMatrix<T, R, C> {
    /// Creates a zero `R × C` matrix.
    #[inline]
    pub fn zeros() -> Self {
        SMatrix {
            matrix: DMatrix::zeros(R, C),
        }
    }

    /// Builds an `R × C` matrix from a row-major slice (matches `nalgebra::SMatrix::from_row_slice`).
    #[inline]
    pub fn from_row_slice(slice: &[T]) -> Self {
        assert_eq!(R * C, slice.len(), "from_row_slice: length mismatch");
        SMatrix {
            matrix: DMatrix::from_row_slice(R, C, slice),
        }
    }

    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        R
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        C
    }
}

impl<T: ComplexField<RealField = T> + Copy, const R: usize, const C: usize> Deref
    for SMatrix<T, R, C>
{
    type Target = DMatrix<T>;
    #[inline]
    fn deref(&self) -> &DMatrix<T> {
        &self.matrix
    }
}

impl<T: ComplexField<RealField = T> + Copy, const R: usize, const C: usize> DerefMut
    for SMatrix<T, R, C>
{
    #[inline]
    fn deref_mut(&mut self) -> &mut DMatrix<T> {
        &mut self.matrix
    }
}

/// A matrix with a **dynamic row count** and dynamic column count, column-major with stride.
///
/// This is the `nalgebra::OMatrix<Real, R, Dyn>` replacement (`R` supplied at runtime, not as a
/// type param, so `gemm` between differently-sized matrices works). It is just a `DMatrix`.
pub type OMatrix<T> = DMatrix<T>;

/// A constraint Jacobian: a matrix with **dynamic row count** (`SPATIAL_DIM` for the spatial
/// twist, but supplied at runtime) and **dynamic columns** (one per generalized DoF).
///
/// This is the `nalgebra::MatrixNxX<N>` (`na::Matrix3xX` / `na::Matrix6xX`) replacement. It is
/// simply `DMatrix<N>` (dynamic nrows, dynamic ncols, column stride = nrows), identical to
/// nalgebra's layout.
pub type Jacobian<N> = DMatrix<N>;

/// Immutable view into a column-major sub-matrix (full or strided).
#[derive(Clone, Copy, Debug)]
pub struct MatrixView<'a, T: ComplexField<RealField = T> + Copy> {
    /// Borrowed column-major slice.
    pub data: &'a [T],
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Distance (in elements) between element `(i, j)` and `(i, j+1)`.
    pub col_stride: usize,
}

impl<'a, T: ComplexField<RealField = T> + Copy> MatrixView<'a, T> {
    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns a copy of column `j`.
    #[inline]
    pub fn column(&self, j: usize) -> DVector<T> {
        let start = j * self.col_stride;
        DVector {
            data: self.data[start..start + self.nrows].to_vec(),
        }
    }

    /// Returns the rows `first..first+len` as a `DVector` view.
    #[inline]
    pub fn rows(&self, first: usize, len: usize) -> DVectorView<'_, T> {
        DVectorView {
            data: &self.data[first..first + len],
        }
    }

    /// Returns the `L` rows starting at row `first` (shifting the column-major data by `first`),
    /// column stride inherited from this view. Matches nalgebra's `fixed_rows::<L>(first)`.
    #[inline]
    pub fn fixed_rows<const L: usize>(&self, first: usize) -> MatrixView<'_, T> {
        let ncols = self.ncols;
        MatrixView {
            data: &self.data[first..first + L + (ncols - 1) * self.col_stride],
            nrows: L,
            ncols,
            col_stride: self.col_stride,
        }
    }

    /// Returns a dense owned copy (compact, column stride = `nrows`).
    #[inline]
    pub fn into_owned(&self) -> DMatrix<T> {
        let mut data = Vec::with_capacity(self.nrows * self.ncols);
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                data.push(self.data[i + j * self.col_stride]);
            }
        }
        DMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
            col_stride: self.nrows,
        }
    }
}

impl<'a, T: ComplexField<RealField = T> + Copy> std::ops::Index<(usize, usize)> for MatrixView<'a, T> {
    type Output = T;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.data[i + j * self.col_stride]
    }
}

impl<'a, T: ComplexField<RealField = T> + Copy> std::ops::IndexMut<(usize, usize)>
    for MatrixViewMut<'a, T>
{
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        &mut self.data[i + j * self.col_stride]
    }
}

impl<'a, T: ComplexField<RealField = T> + Copy> std::ops::Index<(usize, usize)> for MatrixViewMut<'a, T> {
    type Output = T;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.data[i + j * self.col_stride]
    }
}

/// Mutable view into a column-major sub-matrix (full or strided).
#[derive(Debug)]
pub struct MatrixViewMut<'a, T: ComplexField<RealField = T> + Copy> {
    /// Borrowed column-major slice.
    pub data: &'a mut [T],
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Distance (in elements) between element `(i, j)` and `(i, j+1)`.
    pub col_stride: usize,
}

impl<'a, T: ComplexField<RealField = T> + Copy> MatrixViewMut<'a, T> {
    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Element read.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> T {
        self.data[i + j * self.col_stride]
    }

    /// Returns a mutable view of column `j`.
    #[inline]
    pub fn column_mut(&mut self, j: usize) -> DVectorViewMut<'_, T> {
        let start = j * self.col_stride;
        DVectorViewMut {
            data: &mut self.data[start..start + self.nrows],
        }
    }

    /// Mutable `L` rows block starting at `first` (column stride inherited). Matches
    /// nalgebra's `fixed_rows_mut::<L>(first)`.
    #[inline]
    pub fn fixed_rows_mut<const L: usize>(&mut self, first: usize) -> MatrixViewMut<'_, T> {
        let ncols = self.ncols;
        MatrixViewMut {
            data: &mut self.data[first..first + L + (ncols - 1) * self.col_stride],
            nrows: L,
            ncols,
            col_stride: self.col_stride,
        }
    }

    /// Returns a dense owned copy (compact, column stride = `nrows`).
    #[inline]
    pub fn into_owned(&self) -> DMatrix<T> {
        let mut data = Vec::with_capacity(self.nrows * self.ncols);
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                data.push(self.data[i + j * self.col_stride]);
            }
        }
        DMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
            col_stride: self.nrows,
        }
    }

    /// Copies the content of `other` (same dimensions) into `self`.
    #[inline]
    pub fn copy_from(&mut self, other: &MatrixView<'_, T>) {
        assert_eq!(
            (self.nrows, self.ncols),
            (other.nrows, other.ncols),
            "copy_from dimension mismatch"
        );
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                self.data[i + j * self.col_stride] = other.data[i + j * other.col_stride];
            }
        }
    }

    /// Copies the content of `other` (same dimensions) into `self`.
    #[inline]
    pub fn copy_from_dmatrix(&mut self, other: &DMatrix<T>) {
        assert_eq!(
            (self.nrows, self.ncols),
            (other.nrows, other.ncols),
            "copy_from dimension mismatch"
        );
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                self.data[i + j * self.col_stride] = other.data[i + j * other.col_stride];
            }
        }
    }

    /// Fills every element with `value`.
    #[inline]
    pub fn fill(&mut self, value: T) {
        for e in self.data.iter_mut() {
            *e = value;
        }
    }

    /// Element-wise add-assign `rhs` into `self`, matching nalgebra's `zip_apply(o, x, |o, x| *o = x)`.
    #[inline]
    pub fn zip_apply(&mut self, rhs: &MatrixView<'_, T>, mut f: impl FnMut(&mut T, T)) {
        assert_eq!(
            (self.nrows, self.ncols, self.col_stride),
            (rhs.nrows, rhs.ncols, rhs.col_stride),
            "zip_apply dimension mismatch"
        );
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                let di = i + j * self.col_stride;
                let si = i + j * rhs.col_stride;
                f(&mut self.data[di], rhs.data[si]);
            }
        }
    }

    /// Swaps elements `(i0, j0)` and `(i1, j1)` in place.
    #[inline]
    pub fn swap(&mut self, (i0, j0): (usize, usize), (i1, j1): (usize, usize)) {
        self.data
            .swap(i0 + j0 * self.col_stride, i1 + j1 * self.col_stride);
    }

    /// Returns a mutable view of rows `first..first+n` (as a row-block of this view).
    #[inline]
    pub fn rows_mut(&mut self, first: usize, n: usize) -> MatrixViewMut<'_, T> {
        MatrixViewMut {
            data: &mut self.data[first..first + n + (self.ncols - 1) * self.col_stride],
            nrows: n,
            ncols: self.ncols,
            col_stride: self.col_stride,
        }
    }

    /// Computes `self = alpha * a * b + beta * self`, bit-identical to `nalgebra::gemm`.
    #[inline]
    pub fn gemm<Ma: MatrixLike<T>, Mb: MatrixLike<T>>(
        &mut self,
        alpha: T,
        a: &Ma,
        b: &Mb,
        beta: T,
    ) {
        let nrows1 = self.nrows;
        let ncols1 = self.ncols;
        let nrows2 = a.nrows();
        let ncols2 = a.ncols();
        let nrows3 = b.nrows();
        let ncols3 = b.ncols();

        assert_eq!(
            ncols2, nrows3,
            "gemm: dimensions mismatch for multiplication."
        );
        assert_eq!(
            (nrows1, ncols1),
            (nrows2, ncols3),
            "gemm: dimensions mismatch for addition."
        );

        const SMALL_DIM: usize = 5;
        let large =
            nrows1 > SMALL_DIM && ncols1 > SMALL_DIM && nrows2 > SMALL_DIM && ncols2 > SMALL_DIM;

        if large {
            let (rsa, csa) = a.strides();
            let (rsb, csb) = b.strides();
            let (rsc, csc) = (1usize, self.col_stride);
            unsafe {
                matrixmultiply::dgemm(
                    nrows2,
                    ncols2,
                    ncols3,
                    std::mem::transmute_copy(&alpha),
                    a.as_ptr() as *const f64,
                    rsa as isize,
                    csa as isize,
                    b.as_ptr() as *const f64,
                    rsb as isize,
                    csb as isize,
                    std::mem::transmute_copy(&beta),
                    self.data.as_mut_ptr() as *mut f64,
                    rsc as isize,
                    csc as isize,
                );
            }
            return;
        }

        for j1 in 0..ncols1 {
            let mut y = self.column_mut(j1);
            let bcol = b.col(j1);
            gemv_column(&mut y, alpha, a, &bcol.as_view(), beta);
        }
    }

    /// Rank-1 update `self += alpha * x * y^T + beta * self`, bit-identical to `nalgebra::ger`.
    /// `x` has `self.nrows` elements, `y` has `self.ncols` elements.
    #[inline]
    pub fn ger(&mut self, alpha: T, x: &DVector<T>, y: &DVector<T>, beta: T) {
        assert_eq!(self.nrows, x.len(), "ger: x dimension mismatch");
        assert_eq!(self.ncols, y.len(), "ger: y dimension mismatch");
        for j in 0..self.ncols {
            let val = alpha * y.data[j];
            let mut col = self.column_mut(j);
            col.axpy(val, x, beta);
        }
    }

    /// Computes `self = alpha * rhs.transpose() * mid * rhs + beta * self` (quadratic form),
    /// bit-identical to `nalgebra::quadform`.
    #[inline]
    pub fn quadform<M: MatrixLike<T>>(&mut self, alpha: T, mid: &M, rhs: &M, beta: T) {
        let dim = mid.nrows();
        assert_eq!(mid.ncols(), dim, "quadform: mid must be square");
        assert_eq!(rhs.nrows(), dim, "quadform: rhs rows must match mid");
        assert_eq!(
            self.nrows,
            rhs.ncols(),
            "quadform: self rows must match rhs cols"
        );
        assert_eq!(
            self.ncols,
            rhs.ncols(),
            "quadform: self cols must match rhs cols"
        );

        let mut work = DVector::zeros(dim);
        for j in 0..rhs.ncols() {
            work.gemv(T::one(), mid, &rhs.col(j).as_view(), T::zero());
            self.column_mut(j)
                .gemv_tr(alpha, rhs, &work.as_view(), if j == 0 { beta } else { T::one() });
        }
    }

    /// Equivalent to `self.transpose() * rhs` but stores the result (vector) into `out`.
    /// Bit-identical to `nalgebra::tr_mul_to` for the matrix × vector case.
    #[inline]
    pub fn tr_mul_to(&self, rhs: &DVectorView<T>, out: &mut DVector<T>) {
        assert_eq!(self.nrows, rhs.data.len(), "tr_mul_to: row mismatch");
        assert_eq!(self.ncols, out.len(), "tr_mul_to: out mismatch");
        out.gemv_tr(T::one(), self, rhs, T::zero());
    }

    /// Computes `self = alpha * a.transpose() * b + beta * self`, bit-identical to
    /// `nalgebra::gemm_tr`.
    #[inline]
    pub fn gemm_tr<Ma: MatrixLike<T>, Mb: MatrixLike<T>>(
        &mut self,
        alpha: T,
        a: &Ma,
        b: &Mb,
        beta: T,
    ) {
        let nrows1 = self.nrows;
        let ncols1 = self.ncols;
        let nrows2 = a.nrows();
        let ncols2 = a.ncols();
        let nrows3 = b.nrows();
        let ncols3 = b.ncols();

        assert_eq!(
            nrows2, nrows3,
            "gemm_tr: dimensions mismatch for multiplication."
        );
        assert_eq!(
            (nrows1, ncols1),
            (ncols2, ncols3),
            "gemm_tr: dimensions mismatch for addition."
        );

        for j1 in 0..ncols1 {
            let mut y = self.column_mut(j1);
            let bcol = b.col(j1);
            gemv_tr_column(&mut y, alpha, a, &bcol.as_view(), beta);
        }
    }
}

/// Trait abstracting "a matrix I can read columns from with a column stride", so `gemv`/
/// `gemv_tr`/`gemm` work uniformly over `DMatrix`/`OMatrix`/`Jacobian`/`SMatrix`/`MatrixView`/
/// `MatrixViewMut` (mirrors nalgebra's `Storage`).
pub trait MatrixLike<T: ComplexField<RealField = T> + Copy> {
    /// Number of rows.
    fn nrows(&self) -> usize;
    /// Number of columns.
    fn ncols(&self) -> usize;
    /// Returns column `j` as an owned `DVector`.
    fn col(&self, j: usize) -> DVector<T>;
    /// Raw column-major pointer (for `matrixmultiply`).
    fn as_ptr(&self) -> *const T;
    /// Raw strides `(row_stride, col_stride)` (for `matrixmultiply`).
    fn strides(&self) -> (usize, usize);
}

impl<T: ComplexField<RealField = T> + Copy> MatrixLike<T> for DMatrix<T> {
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows
    }
    #[inline]
    fn ncols(&self) -> usize {
        self.ncols
    }
    #[inline]
    fn col(&self, j: usize) -> DVector<T> {
        self.column(j)
    }
    #[inline]
    fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    #[inline]
    fn strides(&self) -> (usize, usize) {
        (1, self.col_stride)
    }
}

impl<T: ComplexField<RealField = T> + Copy> MatrixLike<T> for MatrixView<'_, T> {
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows
    }
    #[inline]
    fn ncols(&self) -> usize {
        self.ncols
    }
    #[inline]
    fn col(&self, j: usize) -> DVector<T> {
        self.column(j)
    }
    #[inline]
    fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    #[inline]
    fn strides(&self) -> (usize, usize) {
        (1, self.col_stride)
    }
}

impl<T: ComplexField<RealField = T> + Copy> MatrixLike<T> for MatrixViewMut<'_, T> {
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows
    }
    #[inline]
    fn ncols(&self) -> usize {
        self.ncols
    }
    #[inline]
    fn col(&self, j: usize) -> DVector<T> {
        let start = j * self.col_stride;
        DVector {
            data: self.data[start..start + self.nrows].to_vec(),
        }
    }
    #[inline]
    fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    #[inline]
    fn strides(&self) -> (usize, usize) {
        (1, self.col_stride)
    }
}

impl<T: ComplexField<RealField = T> + Copy, const R: usize, const C: usize> MatrixLike<T>
    for SMatrix<T, R, C>
{
    #[inline]
    fn nrows(&self) -> usize {
        R
    }
    #[inline]
    fn ncols(&self) -> usize {
        C
    }
    #[inline]
    fn col(&self, j: usize) -> DVector<T> {
        self.column(j)
    }
    #[inline]
    fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    #[inline]
    fn strides(&self) -> (usize, usize) {
        (1, R)
    }
}

/// `ColAccess` keeps `DVector::gemv`/`gemv_tr` working (it only needs `nrows`/`ncols`/`col`).
pub trait ColAccess<T: ComplexField<RealField = T> + Copy> {
    /// Number of rows.
    fn nrows(&self) -> usize;
    /// Number of columns.
    fn ncols(&self) -> usize;
    /// Returns column `j` as an owned `DVector`.
    fn col(&self, j: usize) -> DVector<T>;
}

impl<T: ComplexField<RealField = T> + Copy, M: MatrixLike<T>> ColAccess<T> for M {
    #[inline]
    fn nrows(&self) -> usize {
        MatrixLike::nrows(self)
    }
    #[inline]
    fn ncols(&self) -> usize {
        MatrixLike::ncols(self)
    }
    #[inline]
    fn col(&self, j: usize) -> DVector<T> {
        MatrixLike::col(self, j)
    }
}

/// `y = a * x * c + b * y`.
/// If `b` is zero, `y` is never read from (matching nalgebra's `array_axc`/`array_axcpy`).
#[inline]
fn axpy_col<T: ComplexField<RealField = T> + Copy>(y: &mut [T], alpha: T, x: &[T], c: T, b: T) {
    if !b.is_zero() {
        for i in 0..x.len() {
            y[i] = alpha * x[i] * c + b * y[i];
        }
    } else {
        for i in 0..x.len() {
            y[i] = alpha * x[i] * c;
        }
    }
}

/// `y = alpha * a * x + beta * y`, matching nalgebra's `gemv_uninit`/`axcpy_uninit`.
#[inline]
fn gemv_column<T: ComplexField<RealField = T> + Copy, M: MatrixLike<T>>(
    y: &mut DVectorViewMut<'_, T>,
    alpha: T,
    a: &M,
    x: &DVectorView<T>,
    beta: T,
) {
    let ncols2 = a.ncols();
    assert_eq!(a.nrows(), y.data.len(), "gemv: row mismatch");
    assert_eq!(ncols2, x.data.len(), "gemv: col mismatch");

    for j in 0..ncols2 {
        let a_col = a.col(j);
        let c = x.data[j];
        // y = alpha * a_col * c + (j==0 ? beta : 1) * y   (axcpy semantics)
        let b = if j == 0 { beta } else { T::one() };
        axpy_col(y.data, alpha, &a_col.data, c, b);
    }
}

/// `y = alpha * a.transpose() * x + beta * y`, matching nalgebra's `gemv_tr_uninit`.
#[inline]
fn gemv_tr_column<T: ComplexField<RealField = T> + Copy, M: MatrixLike<T>>(
    y: &mut DVectorViewMut<'_, T>,
    alpha: T,
    a: &M,
    x: &DVectorView<T>,
    beta: T,
) {
    let ncols2 = a.ncols();
    assert_eq!(a.nrows(), x.data.len(), "gemv_tr: row mismatch");
    assert_eq!(ncols2, y.data.len(), "gemv_tr: col mismatch");

    for j in 0..ncols2 {
        let a_col = a.col(j);
        let d = a_col.dot(&DVector::from_slice(x.data));
        if !beta.is_zero() {
            y.data[j] = alpha * d + beta * y.data[j];
        } else {
            y.data[j] = alpha * d;
        }
    }
}

/// A sequence of row permutations (matching `nalgebra::PermutationSequence` semantics).
///
/// Stores at most `len` recorded swaps `(i, i2)`; `permute_rows` applies them in order,
/// `determinant` is `+1` for an even number of swaps and `-1` otherwise.
#[derive(Clone, Debug, Default)]
pub struct PermutationSequence {
    len: usize,
    ipiv: Vec<(usize, usize)>,
}

impl PermutationSequence {
    /// Creates an empty (identity) permutation sequence of capacity `n`.
    #[inline]
    pub fn identity(n: usize) -> Self {
        PermutationSequence {
            len: 0,
            ipiv: vec![(0usize, 0usize); n],
        }
    }

    /// Records the interchange of rows `i` and `i2` (no-op if `i == i2`).
    #[inline]
    pub fn append_permutation(&mut self, i: usize, i2: usize) {
        if i != i2 {
            assert!(
                self.len < self.ipiv.len(),
                "Maximum number of permutations exceeded."
            );
            self.ipiv[self.len] = (i, i2);
            self.len += 1;
        }
    }

    /// Applies this sequence of permutations to the rows of `rhs` (in recorded order).
    #[inline]
    pub fn permute_rows<T: ComplexField<RealField = T> + Copy + PartialOrd>(
        &self,
        rhs: &mut MatrixViewMut<'_, T>,
    ) {
        for k in 0..self.len {
            let (i0, i1) = self.ipiv[k];
            for j in 0..rhs.ncols {
                rhs.swap((i0, j), (i1, j));
            }
        }
    }

    /// The determinant contribution of this permutation (+1 even / -1 odd count).
    #[inline]
    pub fn determinant<T: ComplexField<RealField = T> + Copy>(&self) -> T {
        if self.len % 2 == 0 {
            T::one()
        } else {
            -T::one()
        }
    }
}

/// LU decomposition with partial (row) pivoting (`nalgebra::LU` replacement).
///
/// Stored as the combined `lu` matrix (L's strictly-lower part overwritten with U's multipliers,
/// matching `nalgebra`'s in-place layout) plus the row-permutation sequence `p`. The decomposition
/// arithmetic is a **verbatim** copy of `nalgebra`'s `linalg/lu.rs` so results are bit-identical.
#[derive(Clone, Debug)]
pub struct LU<T: ComplexField<RealField = T> + Copy> {
    /// Combined lower/upper factor (in-place, column-major).
    pub lu: DMatrix<T>,
    /// Row permutation sequence.
    pub p: PermutationSequence,
}

/// A right-hand side that an `LU` decomposition can solve into (vector or matrix view).
pub trait SolveTarget<T: ComplexField<RealField = T> + Copy> {
    /// Exposes the target as a mutable column-major view for in-place solve.
    fn as_view_mut(&mut self) -> MatrixViewMut<'_, T>;
}

impl<T: ComplexField<RealField = T> + Copy> SolveTarget<T> for DMatrix<T> {
    #[inline]
    fn as_view_mut(&mut self) -> MatrixViewMut<'_, T> {
        MatrixViewMut {
            data: &mut self.data,
            nrows: self.nrows,
            ncols: self.ncols,
            col_stride: self.col_stride,
        }
    }
}

impl<T: ComplexField<RealField = T> + Copy> SolveTarget<T> for DVector<T> {
    #[inline]
    fn as_view_mut(&mut self) -> MatrixViewMut<'_, T> {
        let n = self.data.len();
        MatrixViewMut {
            data: &mut self.data,
            nrows: n,
            ncols: 1,
            col_stride: n,
        }
    }
}

impl<'a, T: ComplexField<RealField = T> + Copy> SolveTarget<T> for DVectorViewMut<'a, T> {
    #[inline]
    fn as_view_mut(&mut self) -> MatrixViewMut<'_, T> {
        let n = self.data.len();
        MatrixViewMut {
            data: self.data,
            nrows: n,
            ncols: 1,
            col_stride: n,
        }
    }
}

impl<'a, T: ComplexField<RealField = T> + Copy> SolveTarget<T> for MatrixViewMut<'a, T> {
    #[inline]
    fn as_view_mut(&mut self) -> MatrixViewMut<'_, T> {
        MatrixViewMut {
            data: self.data,
            nrows: self.nrows,
            ncols: self.ncols,
            col_stride: self.col_stride,
        }
    }
}

impl<T: ComplexField<RealField = T> + Copy + PartialOrd> LU<T> {
    /// Computes the LU decomposition with partial (row) pivoting of `matrix`.
    pub fn new(mut matrix: DMatrix<T>) -> Self {
        let min_n = matrix.nrows.min(matrix.ncols);
        let mut p = PermutationSequence::identity(min_n);

        if min_n == 0 {
            return LU { lu: matrix, p };
        }

        for i in 0..min_n {
            let piv = matrix.icamax(i, i);
            let diag = matrix.get(piv, i);

            if diag.is_zero() {
                // No non-zero entry on this column; leave the row as-is.
                continue;
            }

            if piv != i {
                p.append_permutation(i, piv);
                // Swap rows `i` and `piv` in the already-processed columns `..i`
                // (nalgebra: `matrix.columns_range_mut(..i).swap_rows(i, piv)`).
                for j in 0..i {
                    matrix.swap((i, j), (piv, j));
                }
                gauss_step_swap(&mut matrix, diag, i, piv);
            } else {
                gauss_step(&mut matrix, diag, i);
            }
        }

        LU { lu: matrix, p }
    }

    /// Solves `self * x = b` in place. Returns `false` if the matrix is not invertible.
    pub fn solve_mut<B: SolveTarget<T>>(&self, b: &mut B) -> bool {
        let mut bv = b.as_view_mut();
        assert_eq!(
            self.lu.nrows, bv.nrows,
            "LU solve matrix dimension mismatch."
        );
        assert!(
            self.lu.is_square(),
            "LU solve: unable to solve a non-square system."
        );

        self.p.permute_rows(&mut bv);
        let _ = solve_lower_triangular_with_diag_mut(&self.lu, &mut bv, T::one());
        solve_upper_triangular_mut(&self.lu, &mut bv)
    }

    /// Solves `self * x = b`, returning `None` if not invertible.
    pub fn solve(&self, b: &DMatrix<T>) -> Option<DMatrix<T>> {
        let mut res = b.clone_owned();
        if self.solve_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Computes the inverse of the decomposed matrix. Returns `None` if not invertible.
    pub fn try_inverse(&self) -> Option<DMatrix<T>> {
        assert!(
            self.lu.is_square(),
            "LU inverse: unable to compute the inverse of a non-square matrix."
        );
        let dim = self.lu.nrows;
        let mut res = DMatrix::zeros(dim, dim);
        res.fill_with_identity();
        if self.try_inverse_to(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Computes the inverse into `out`. Returns `false` if not invertible.
    pub fn try_inverse_to(&self, out: &mut DMatrix<T>) -> bool {
        assert!(
            self.lu.is_square(),
            "LU inverse: unable to compute the inverse of a non-square matrix."
        );
        assert_eq!(
            self.lu.shape(),
            out.shape(),
            "LU inverse: mismatched output shape."
        );
        out.fill_with_identity();
        self.solve_mut(out)
    }

    /// The determinant of the decomposed matrix.
    pub fn to_determinant(&self) -> T {
        let dim = self.lu.nrows;
        assert!(
            self.lu.is_square(),
            "LU determinant: unable to compute the determinant of a non-square matrix."
        );
        let mut res = T::one();
        for i in 0..dim {
            res *= self.lu.get(i, i);
        }
        res * self.p.determinant()
    }
}

impl<T: ComplexField<RealField = T> + Copy> DMatrix<T> {
    /// Sets `self` to the identity matrix (diagonal `1`, off-diagonal `0`).
    #[inline]
    pub fn fill_with_identity(&mut self) {
        self.fill(T::zero());
        let d = self.nrows.min(self.ncols);
        for i in 0..d {
            self.data[i + i * self.col_stride] = T::one();
        }
    }

    /// Returns `(nrows, ncols)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
}

/// One Gaussian-elimination step on the i-th row/column (no row swap). The diagonal `diag` is
/// provided. Verbatim mirror of `nalgebra`'s `gauss_step`: `coeffs` (the pivot column, rows 1..)
/// is scaled by `inv_diag` in place (storing the L multipliers in column `i`), then every column
/// to the right is eliminated against it.
pub fn gauss_step<T: ComplexField<RealField = T> + Copy>(
    matrix: &mut DMatrix<T>,
    diag: T,
    i: usize,
) {
    let nrows = matrix.nrows;
    let ncols = matrix.ncols;
    let inv_diag = T::one() / diag;

    // Store the L multipliers `matrix[(r, i)] / diag` back into column `i` (nalgebra's
    // `coeffs *= inv_diag` mutates the underlying storage in place).
    for r in 1..(nrows - i) {
        let v = matrix.get(i + r, i) * inv_diag;
        *matrix.get_mut(i + r, i) = v;
    }

    for k in 1..(ncols - i) {
        let pj = matrix.get(i, i + k);
        for r in 1..(nrows - i) {
            let cr = matrix.get(i + r, i);
            let cur = matrix.get(i + r, i + k);
            matrix.data[(i + r) + (i + k) * matrix.col_stride] = -pj * cr + cur;
        }
    }
}

/// Gaussian-elimination step with a prior row swap (`piv` is the absolute pivot row). Verbatim
/// mirror of `nalgebra`'s `gauss_step_swap`.
pub fn gauss_step_swap<T: ComplexField<RealField = T> + Copy>(
    matrix: &mut DMatrix<T>,
    diag: T,
    i: usize,
    piv: usize,
) {
    let nrows = matrix.nrows;
    let ncols = matrix.ncols;
    let inv_diag = T::one() / diag;

    // coeffs.swap((0, 0), (pk, 0)) on the submatrix (which starts at row `i`): `pk = piv - i`
    // is the local pivot, so submatrix row `pk` is matrix row `piv` (absolute). Swaps
    // matrix[(i, i)] (the pivot element) with matrix[(piv, i)].
    matrix.swap((i, i), (piv, i));

    // `coeffs *= inv_diag` stores the L multipliers into column `i` (rows 1..).
    for r in 1..(nrows - i) {
        let v = matrix.get(i + r, i) * inv_diag;
        *matrix.get_mut(i + r, i) = v;
    }

    // pivot_row[k] <-> down[(pk-1, k)] for columns `i+1 .. ncols` (column `i` is `coeffs`,
    // already handled above — do NOT touch it here). `down[(pk-1, k)]` is submatrix row `pk`
    // = matrix row `piv` (absolute).
    for k in 0..(ncols - i - 1) {
        matrix.swap((i, i + 1 + k), (piv, i + 1 + k));
    }

    for k in 0..(ncols - i - 1) {
        let pj = matrix.get(i, i + 1 + k);
        for r in 1..(nrows - i) {
            let cr = matrix.get(i + r, i);
            let cur = matrix.get(i + r, i + 1 + k);
            matrix.data[(i + r) + (i + 1 + k) * matrix.col_stride] = -pj * cr + cur;
        }
    }
}

/// Solves `self * x = b` where only the lower-triangular part (with the given `diag`) is used.
/// Returns `false` if `diag` is zero. Verbatim mirror of `nalgebra`'s
/// `solve_lower_triangular_with_diag_mut`.
pub fn solve_lower_triangular_with_diag_mut<T: ComplexField<RealField = T> + Copy>(
    lu: &DMatrix<T>,
    b: &mut MatrixViewMut<'_, T>,
    diag: T,
) -> bool {
    if diag.is_zero() {
        return false;
    }
    let dim = lu.nrows;
    let cols = b.ncols;
    for k in 0..cols {
        for i in 0..dim - 1 {
            let coeff = b.get(i, k) / diag;
            for r in (i + 1)..dim {
                let pivot = lu.get(r, i);
                let cur = b.get(r, k);
                b.data[r + k * b.col_stride] = cur - coeff * pivot;
            }
        }
    }
    true
}

/// Solves `self * x = b` where only the upper-triangular part (incl. diagonal) is used.
/// Returns `false` if any diagonal element is zero. Verbatim mirror of `nalgebra`'s
/// `solve_upper_triangular_mut`.
pub fn solve_upper_triangular_mut<T: ComplexField<RealField = T> + Copy>(
    lu: &DMatrix<T>,
    b: &mut MatrixViewMut<'_, T>,
) -> bool {
    let dim = lu.nrows;
    let cols = b.ncols;
    for k in 0..cols {
        for i in (0..dim).rev() {
            let d = lu.get(i, i);
            if d.is_zero() {
                return false;
            }
            let coeff = b.get(i, k) / d;
            b.data[i + k * b.col_stride] = coeff;
            for r in 0..i {
                let pivot = lu.get(r, i);
                let cur = b.get(r, k);
                b.data[r + k * b.col_stride] = cur - coeff * pivot;
            }
        }
    }
    true
}

impl<T: ComplexField<RealField = T> + Copy + PartialOrd> DMatrix<T> {
    /// Index of the element with the largest absolute value in column `j` starting at row `r0`,
    /// returned as the absolute row index (matching nalgebra's `view_range(r0.., j).icamax()`).
    /// Strict `>` so the **first** maximal element wins (matching nalgebra's `icamax`).
    fn icamax(&self, r0: usize, j: usize) -> usize {
        let mut best_row = r0;
        let mut best = self.get(r0, j).abs();
        for k in (r0 + 1)..self.nrows {
            let v = self.get(k, j).abs();
            if v > best {
                best = v;
                best_row = k;
            }
        }
        best_row
    }
}

impl<T: ComplexField<RealField = T> + Copy> std::ops::Add for DMatrix<T> {
    type Output = DMatrix<T>;
    #[inline]
    fn add(mut self, rhs: DMatrix<T>) -> DMatrix<T> {
        self += rhs;
        self
    }
}

impl<T: ComplexField<RealField = T> + Copy> std::ops::AddAssign for DMatrix<T> {
    #[inline]
    fn add_assign(&mut self, rhs: DMatrix<T>) {
        assert_eq!(
            (self.nrows, self.ncols, self.col_stride),
            (rhs.nrows, rhs.ncols, rhs.col_stride),
            "AddAssign dimension/stride mismatch"
        );
        for i in 0..self.data.len() {
            self.data[i] += rhs.data[i];
        }
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
            assert!(
                approx_eq_bits(m, nref),
                "dot n={} mismatch: {} vs {}",
                n,
                m,
                nref
            );
        }
    }

    #[test]
    fn dvector_component_mul_bit_identical() {
        let a: Vec<f64> = (0..100).map(|i| (i as f64) * 0.91 + 0.1).collect();
        let b: Vec<f64> = (0..100).map(|i| (i as f64) * 0.37 - 0.4).collect();
        let m = DVector::from_vec(a.clone()).component_mul(&DVector::from_vec(b.clone()));
        let nref = NaDVector::from_vec(a).component_mul(&NaDVector::from_vec(b));
        for (x, y) in m.data.iter().zip(nref.iter()) {
            assert!(
                approx_eq_bits(*x, *y),
                "component_mul mismatch {} vs {}",
                x,
                y
            );
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

    #[test]
    fn dmatrix_gemm_bit_identical() {
        use nalgebra::DMatrix as NaDMatrix;

        // exercise both paths: large (matrixmultiply) and small (gemv loop)
        for (nr, nc, mk) in [
            (10usize, 10, 10),
            (4, 4, 4),
            (7, 3, 5),
            (6, 6, 6),
            (20, 20, 20),
        ] {
            let a: Vec<f64> = (0..nr * mk).map(|i| (i as f64) * 0.137 + 0.5).collect();
            let b: Vec<f64> = (0..mk * nc).map(|i| (i as f64) * 0.731 - 0.3).collect();

            for (alpha, beta) in [(1.0, 0.0), (2.0, 1.0), (1.0, 1.0), (0.5, -1.0)] {
                // C starts zero (beta=0 path) or random (beta!=0 path)
                let c0: Vec<f64> = (0..nr * nc).map(|i| (i as f64) * 0.21 - 0.7).collect();

                let mut mine = DMatrix::from_row_slice(nr, nc, &c0);
                let na_c0 = c0.clone();
                let mut na_res = NaDMatrix::from_row_slice(nr, nc, &na_c0);
                let na_a = NaDMatrix::from_row_slice(nr, mk, &a);
                let na_b = NaDMatrix::from_row_slice(mk, nc, &b);

                mine.gemm(
                    alpha,
                    &DMatrix::from_row_slice(nr, mk, &a),
                    &DMatrix::from_row_slice(mk, nc, &b),
                    beta,
                );
                na_res.gemm(alpha, &na_a, &na_b, beta);

                for (m, n) in mine.data.iter().zip(na_res.iter()) {
                    assert!(
                        approx_eq_bits(*m, *n),
                        "gemm ({},{},{}) a={}b={} mismatch: {} vs {}",
                        nr,
                        nc,
                        mk,
                        alpha,
                        beta,
                        m,
                        n
                    );
                }
            }
        }
    }

    #[test]
    fn dmatrix_transpose_bit_identical() {
        use nalgebra::DMatrix as NaDMatrix;
        let (nr, nc) = (6usize, 4);
        let a: Vec<f64> = (0..nr * nc).map(|i| (i as f64) * 0.137 + 0.5).collect();
        let mine = DMatrix::from_row_slice(nr, nc, &a).transpose();
        let na = NaDMatrix::from_row_slice(nr, nc, &a).transpose();
        for (m, n) in mine.data.iter().zip(na.iter()) {
            assert!(approx_eq_bits(*m, *n), "transpose mismatch {} vs {}", m, n);
        }
    }

    #[test]
    fn omatrix_gemm_bit_identical() {
        use nalgebra::DMatrix as NaDMatrix;
        let r = 6usize;
        let n = 7usize;
        let c = 5usize;
        let a_data: Vec<f64> = (0..r * n).map(|i| (i as f64) * 0.137 + 0.5).collect();
        let b_data: Vec<f64> = (0..n * c).map(|i| (i as f64) * 0.731 - 0.3).collect();
        let c0: Vec<f64> = (0..r * c).map(|i| (i as f64) * 0.21 - 0.7).collect();

        for (alpha, beta) in [(1.0, 0.0), (2.0, 1.0), (1.0, 1.0)] {
            let mut mine = OMatrix::from_row_slice(r, c, &c0);
            let mut na_res = NaDMatrix::from_row_slice(r, c, &c0);
            let na_a = NaDMatrix::from_row_slice(r, n, &a_data);
            let na_b = NaDMatrix::from_row_slice(n, c, &b_data);
            mine.gemm(
                alpha,
                &OMatrix::from_row_slice(r, n, &a_data),
                &OMatrix::from_row_slice(n, c, &b_data),
                beta,
            );
            na_res.gemm(alpha, &na_a, &na_b, beta);

            for (m, n) in mine.data.iter().zip(na_res.iter()) {
                assert!(
                    approx_eq_bits(*m, *n),
                    "omatrix gemm mismatch {} vs {}",
                    m,
                    n
                );
            }
        }
    }

    #[test]
    fn omatrix_fixed_rows_gemm_bit_identical() {
        // multibody pattern (dim3): link_j_v (3 rows = DIM) =
        //   gcross_matrix_tr(shift02) (3x3) * parent_j_w (ANG_DIM=3 rows of parent Jacobian, N cols)
        // The 3x3 shift_tr comes from `Vector3::gcross_matrix_tr()` (see cross_product_matrix.rs).
        use nalgebra::DMatrix as NaDMatrix;
        let n = 3usize; // shift is 3x3
        let c = 4usize; // ndofs
        let shift: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.11 - 0.3).collect();
        let parent: Vec<f64> = (0..6 * c).map(|i| (i as f64) * 0.731 + 0.5).collect();

        // result: link_j_v (3 rows) = shift_tr (3x3) * parent_j_w (3 cols of parent)
        let mut mine = OMatrix::<f64>::zeros(3, c);
        let shift_tr = OMatrix::<f64>::from_row_slice(n, n, &shift);
        let parent_full = OMatrix::<f64>::from_row_slice(6, c, &parent);
        let parent_j_w = parent_full.fixed_rows::<3>(3);
        mine.gemm(1.0, &shift_tr, &parent_j_w, 1.0);

        // nalgebra oracle (fully dynamic so it type-checks the mixed-row gemm).
        let na_shift = NaDMatrix::from_row_slice(n, n, &shift);
        let na_parent = NaDMatrix::from_row_slice(6, c, &parent);
        let na_pw = na_parent.fixed_rows::<3>(3);
        let mut na_link = NaDMatrix::zeros(3, c);
        na_link.gemm(1.0, &na_shift, &na_pw, 1.0);

        for (m, n) in mine.data.iter().zip(na_link.iter()) {
            assert!(
                approx_eq_bits(*m, *n),
                "fixed_rows gemm mismatch {} vs {}",
                m,
                n
            );
        }
    }

    #[test]
    fn omatrix_gemm_tr_bit_identical() {
        use nalgebra::OMatrix as NaOMatrix;
        let n = 7usize;
        let c = 5usize;
        let a_data: Vec<f64> = (0..6 * n).map(|i| (i as f64) * 0.137 + 0.5).collect();
        let b_data: Vec<f64> = (0..6 * c).map(|i| (i as f64) * 0.731 - 0.3).collect();

        let mut mine = OMatrix::<f64>::zeros(n, c);
        let na_res = NaOMatrix::<f64, nalgebra::Const<7>, nalgebra::Dyn>::zeros(c);
        let na_a = NaOMatrix::<f64, nalgebra::Const<6>, nalgebra::Dyn>::from_row_slice(&a_data);
        let na_b = NaOMatrix::<f64, nalgebra::Const<6>, nalgebra::Dyn>::from_row_slice(&b_data);
        mine.gemm_tr(
            1.5,
            &OMatrix::from_row_slice(6, n, &a_data),
            &OMatrix::from_row_slice(6, c, &b_data),
            2.0,
        );
        let mut na_res = na_res;
        na_res.gemm_tr(1.5, &na_a, &na_b, 2.0);

        for (m, n) in mine.data.iter().zip(na_res.iter()) {
            assert!(
                approx_eq_bits(*m, *n),
                "omatrix gemm_tr mismatch {} vs {}",
                m,
                n
            );
        }
    }

    #[test]
    fn omatrix_quadform_bit_identical() {
        use crate::linalg::DMatrix;
        let r = 6usize;
        let c = 4usize;
        let mid_data: Vec<f64> = (0..r * r).map(|i| (i as f64) * 0.137 + 0.5).collect();
        let rhs_data: Vec<f64> = (0..r * c).map(|i| (i as f64) * 0.731 - 0.3).collect();
        let c0: Vec<f64> = (0..c * c).map(|i| (i as f64) * 0.21 - 0.7).collect();

        let mid = DMatrix::from_row_slice(r, r, &mid_data);
        let rhs = OMatrix::from_row_slice(r, c, &rhs_data);
        let mut mine = OMatrix::from_row_slice(c, c, &c0);
        mine.quadform(1.0, &mid, &rhs, 1.0);

        // oracle via nalgebra's own quadform (bit-identical algorithm)
        let na_mid = nalgebra::DMatrix::<f64>::from_row_slice(r, r, &mid_data);
        let na_rhs = nalgebra::DMatrix::<f64>::from_row_slice(r, c, &rhs_data);
        let mut na_res = nalgebra::DMatrix::<f64>::from_row_slice(c, c, &c0);
        na_res.quadform(1.0, &na_mid, &na_rhs, 1.0);

        for (m, n) in mine.data.iter().zip(na_res.iter()) {
            assert!(
                approx_eq_bits(*m, *n),
                "omatrix quadform mismatch {} vs {}",
                m,
                n
            );
        }
    }

    #[test]
    fn omatrix_ger_bit_identical() {
        use nalgebra::OMatrix as NaOMatrix;
        let r = 2usize;
        let c = 3usize;
        let a_data: Vec<f64> = (0..r * c).map(|i| (i as f64) * 0.137 + 0.5).collect();
        let x: Vec<f64> = (0..r).map(|i| (i as f64) * 0.21 - 0.7).collect();
        let y: Vec<f64> = (0..c).map(|i| (i as f64) * 0.731 + 0.3).collect();

        let mut mine = OMatrix::from_row_slice(r, c, &a_data);
        let mut na_res =
            NaOMatrix::<f64, nalgebra::Const<2>, nalgebra::Const<3>>::from_row_slice(&a_data);
        let na_x = nalgebra::DVector::from_vec(x.clone());
        let na_y = nalgebra::DVector::from_vec(y.clone());
        let xv = DVector::from_vec(x);
        let yv = DVector::from_vec(y);
        mine.ger(1.0, &xv, &yv, 1.0);
        na_res.ger(1.0, &na_x, &na_y, 1.0);

        for (m, n) in mine.data.iter().zip(na_res.iter()) {
            assert!(
                approx_eq_bits(*m, *n),
                "omatrix ger mismatch {} vs {}",
                m,
                n
            );
        }
    }

    #[test]
    fn omatrix_tr_mul_to_bit_identical() {
        use nalgebra::OMatrix as NaOMatrix;
        let r = 6usize;
        let c = 4usize;
        let j: Vec<f64> = (0..r * c).map(|i| (i as f64) * 0.137 + 0.5).collect();
        let f: Vec<f64> = (0..r).map(|i| (i as f64) * 0.731 - 0.3).collect();

        let mine = OMatrix::from_row_slice(r, c, &j);
        let na_j = NaOMatrix::<f64, nalgebra::Const<6>, nalgebra::Dyn>::from_row_slice(&j);
        let na_f = nalgebra::DVector::from_vec(f.clone());
        let mut out = DVector::zeros(c);
        let f_vec = DVector::from_vec(f);
        let fv = f_vec.as_view();
        mine.tr_mul_to(&fv, &mut out);
        let na_out = na_j.tr_mul(&na_f);

        for (m, n) in out.data.iter().zip(na_out.iter()) {
            assert!(approx_eq_bits(*m, *n), "tr_mul_to mismatch {} vs {}", m, n);
        }
    }

    #[test]
    fn smatrix_gemm_bit_identical() {
        // SMatrix (wraps a DMatrix) must gemm bit-identically with nalgebra::SMatrix.
        let a_data: Vec<f64> = (0..3 * 3).map(|i| (i as f64) * 0.137 + 0.5).collect();
        let b_data: Vec<f64> = (0..3 * 3).map(|i| (i as f64) * 0.731 - 0.3).collect();

        let a = SMatrix::<f64, 3, 3>::from_row_slice(&a_data);
        let b = SMatrix::<f64, 3, 3>::from_row_slice(&b_data);
        let mut mine = SMatrix::<f64, 3, 3>::zeros();
        mine.gemm(2.0, &a, &b, 1.0);

        // nalgebra oracle (apples-to-apples: nalgebra::SMatrix<f64,3,3>).
        let na_a = nalgebra::SMatrix::<f64, 3, 3>::from_row_slice(&a_data);
        let na_b = nalgebra::SMatrix::<f64, 3, 3>::from_row_slice(&b_data);
        let mut na_res = nalgebra::SMatrix::<f64, 3, 3>::zeros();
        na_res.gemm(2.0, &na_a, &na_b, 1.0);

        for (m, n) in mine.matrix.data.iter().zip(na_res.iter()) {
            assert!(
                approx_eq_bits(*m, *n),
                "smatrix gemm mismatch {} vs {}",
                m,
                n
            );
        }
    }

    /// Builds a deterministic pseudo-random `nrows × ncols` matrix (no std RNG dependency).
    fn rand_matrix(nrows: usize, ncols: usize, seed: u64) -> DMatrix<f64> {
        let mut data = Vec::with_capacity(nrows * ncols);
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        for _ in 0..(nrows * ncols) {
            // xorshift64
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let v = ((s >> 11) as f64) / ((1u64 << 53) as f64) * 10.0 - 5.0;
            data.push(v);
        }
        DMatrix::from_row_slice(nrows, ncols, &data)
    }

    #[test]
    fn lu_decomposition_bit_identical() {
        for (r, c, seed) in [(3usize, 3, 1u64), (4, 4, 2), (5, 5, 7), (6, 6, 13)] {
            let m = rand_matrix(r, c, seed);
            let mine = LU::new(m.clone_owned());

            let na = nalgebra::LU::<f64, nalgebra::Dyn, nalgebra::Dyn>::new(
                nalgebra::DMatrix::from_column_slice(r, c, &m.data),
            );

            // Compare the decomposed L (unit-diagonal) and U (upper) factors bit-by-bit.
            let na_l = na.l();
            let na_u = na.u();
            for i in 0..r {
                for j in 0..c {
                    let my_l = if i > j {
                        mine.lu.get(i, j)
                    } else if i == j {
                        1.0
                    } else {
                        0.0
                    };
                    let my_u = if i <= j { mine.lu.get(i, j) } else { 0.0 };
                    assert!(
                        approx_eq_bits(my_l, na_l[(i, j)]),
                        "LU L mismatch r={r} c={c} seed={seed} ({i},{j}): {my_l} vs {}",
                        na_l[(i, j)]
                    );
                    assert!(
                        approx_eq_bits(my_u, na_u[(i, j)]),
                        "LU U mismatch r={r} c={c} seed={seed} ({i},{j}): {my_u} vs {}",
                        na_u[(i, j)]
                    );
                }
            }
        }
    }

    #[test]
    fn lu_solve_bit_identical() {
        for (n, seed, bseed) in [(4usize, 3u64, 100), (6, 9, 200), (5, 17, 300)] {
            let a = rand_matrix(n, n, seed);
            let b = rand_matrix(n, 2, bseed);
            let mine = LU::new(a.clone_owned());

            let mut x_mine = b.clone_owned();
            assert!(mine.solve_mut(&mut x_mine));

            let na = nalgebra::LU::<f64, nalgebra::Dyn, nalgebra::Dyn>::new(
                nalgebra::DMatrix::from_column_slice(n, n, &a.data),
            );
            let na_b = nalgebra::DMatrix::from_column_slice(n, 2, &b.data);
            let x_na = na.solve(&na_b).unwrap();

            for i in 0..n {
                for j in 0..2 {
                    assert!(
                        approx_eq_bits(x_mine.get(i, j), x_na[(i, j)]),
                        "LU solve mismatch n={n} ({i},{j}): {} vs {}",
                        x_mine.get(i, j),
                        x_na[(i, j)]
                    );
                }
            }
        }
    }

    #[test]
    fn lu_inverse_bit_identical() {
        for (n, seed) in [(4usize, 5u64), (5, 11), (6, 27)] {
            let a = rand_matrix(n, n, seed);
            let mine = LU::new(a.clone_owned());
            let inv_mine = mine.try_inverse().expect("invertible");

            let na = nalgebra::LU::<f64, nalgebra::Dyn, nalgebra::Dyn>::new(
                nalgebra::DMatrix::from_column_slice(n, n, &a.data),
            );
            let inv_na = na.try_inverse().unwrap();

            for i in 0..n {
                for j in 0..n {
                    assert!(
                        approx_eq_bits(inv_mine.get(i, j), inv_na[(i, j)]),
                        "LU inverse mismatch n={n} ({i},{j}): {} vs {}",
                        inv_mine.get(i, j),
                        inv_na[(i, j)]
                    );
                }
            }
        }
    }

    #[test]
    fn lu_determinant_bit_identical() {
        for (n, seed) in [(3usize, 2u64), (5, 8), (4, 19), (6, 33)] {
            let a = rand_matrix(n, n, seed);
            let mine = LU::new(a.clone_owned());
            let det_mine = mine.to_determinant();

            let na = nalgebra::LU::<f64, nalgebra::Dyn, nalgebra::Dyn>::new(
                nalgebra::DMatrix::from_column_slice(n, n, &a.data),
            );
            let det_na = na.determinant();

            assert!(
                approx_eq_bits(det_mine, det_na),
                "LU determinant mismatch n={n}: {det_mine} vs {det_na}"
            );
        }
    }
}
