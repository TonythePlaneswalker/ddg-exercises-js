"use strict";

class HeatMethod {
	/**
	 * This class implements the {@link http://cs.cmu.edu/~kmcrane/Projects/HeatMethod/ heat method} to compute geodesic distance
	 * on a surface mesh.
	 * @constructor module:Projects.HeatMethod
	 * @param {module:Core.Geometry} geometry The input geometry of the mesh this class acts on.
	 * @property {module:Core.Geometry} geometry The input geometry of the mesh this class acts on.
	 * @property {Object} vertexIndex A dictionary mapping each vertex of the input mesh to a unique index.
	 * @property {module:LinearAlgebra.SparseMatrix} A The laplace matrix of the input mesh.
	 * @property {module:LinearAlgebra.SparseMatrix} F The mean curvature flow operator built on the input mesh.
	 */
	constructor(geometry) {
		this.geometry = geometry;
		this.vertexIndex = indexElements(geometry.mesh.vertices);

		// Build laplace and flow matrices
		this.A = this.geometry.laplaceMatrix(this.vertexIndex);
		this.Allt = this.A.chol();

		let M = this.geometry.massMatrix(this.vertexIndex);
		let h = this.geometry.meanEdgeLength();
		this.F = M.plus(this.A.timesReal(h * h));
		this.Fllt = this.F.chol();
	}

	/**
	 * Computes the vector field X = -∇u / |∇u|.
	 * @private
	 * @method module:Projects.HeatMethod#computeVectorField
	 * @param {module:LinearAlgebra.DenseMatrix} u A dense vector (i.e., u.nCols() == 1) representing the
	 * heat that is allowed to diffuse on the input mesh for a brief period of time.
	 * @returns {Object} A dictionary mapping each face of the input mesh to a {@link module:LinearAlgebra.Vector Vector}.
	 */
	computeVectorField(u) {
		let X = {};
		for (let f of this.geometry.mesh.faces) {
			let du = new Vector();
			let n = this.geometry.faceNormal(f);
			for (let h of f.adjacentHalfedges()) {
				let i = h.prev.vertex.index;
				let e = this.geometry.vector(h);
				du.incrementBy(n.cross(e).times(u.get(i, 0)));
			}
			X[f] = du.unit().negated();
		}
		return X;
	}

	/**
	 * Computes the integrated divergence ∇.X.
	 * @private
	 * @method module:Projects.HeatMethod#computeDivergence
	 * @param {Object} X The vector field -∇u / |∇u| represented by a dictionary
	 * mapping each face of the input mesh to a {@link module:LinearAlgebra.Vector Vector}.
	 * @returns {module:LinearAlgebra.DenseMatrix}
	 */
	computeDivergence(X) {
		let divX = DenseMatrix.zeros(this.geometry.mesh.vertices.length, 1);
		for (let v of this.geometry.mesh.vertices) {
			let s = 0;
			for (let f of v.adjacentFaces()) {
				let h = f.halfedge;
				while (h.vertex != v) {
					h = h.next;
				}
				s += (this.geometry.cotan(h)*this.geometry.vector(h).dot(X[f])
					+ this.geometry.cotan(h.prev)*this.geometry.vector(h.prev.twin).dot(X[f]))
			}
			divX.set(s / 2, v.index, 0);
		}
		return divX;
	}

	/**
	 * Shifts φ such that its minimum value is zero.
	 * @private
	 * @method module:Projects.HeatMethod#subtractMinimumDistance
	 * @param {module:LinearAlgebra.DenseMatrix} phi The (minimum 0) solution to the poisson equation Δφ = ∇.X.
	 */
	subtractMinimumDistance(phi) {
		let min = Infinity;
		for (let i = 0; i < phi.nRows(); i++) {
			min = Math.min(phi.get(i, 0), min);
		}

		for (let i = 0; i < phi.nRows(); i++) {
			phi.set(phi.get(i, 0) - min, i, 0);
		}
	}

	/**
	 * Computes the geodesic distances φ using the heat method.
	 * @method module:Projects.HeatMethod#compute
	 * @param {module:LinearAlgebra.DenseMatrix} delta A dense vector (i.e., delta.nCols() == 1) containing
	 * heat sources, i.e., u0 = δ(x).
	 * @returns {module:LinearAlgebra.DenseMatrix}
	 */
	compute(delta) {
		let u = this.Fllt.solvePositiveDefinite(delta);
		let X = this.computeVectorField(u);
		let divX = this.computeDivergence(X);
		let phi = this.Allt.solvePositiveDefinite(divX.negated());

		// since φ is unique up to an additive constant, it should
		// be shifted such that the smallest distance is zero
		this.subtractMinimumDistance(phi);

		return phi;
	}
}
