"use strict";

class MeanCurvatureFlow {
	/**
	 * This class performs {@link https://cs.cmu.edu/~kmcrane/Projects/DDG/paper.pdf mean curvature flow} on a surface mesh.
	 * @constructor module:Projects.MeanCurvatureFlow
	 * @param {module:Core.Geometry} geometry The input geometry of the mesh this class acts on.
	 * @property {module:Core.Geometry} geometry The input geometry of the mesh this class acts on.
	 * @property {Object} vertexIndex A dictionary mapping each vertex of the input mesh to a unique index.
	 */
	constructor(geometry) {
		this.geometry = geometry;
		this.vertexIndex = indexElements(geometry.mesh.vertices);
	}

	/**
	 * Builds the mean curvature flow operator.
	 * @private
	 * @method module:Projects.MeanCurvatureFlow#buildFlowOperator
	 * @param {module:LinearAlgebra.SparseMatrix} M The mass matrix of the input mesh.
	 * @param {number} h The timestep.
	 * @returns {module:LinearAlgebra.SparseMatrix}
	 */
	buildFlowOperator(M, h) {
		let L = this.geometry.laplaceMatrix(this.vertexIndex);
		return M.plus(L.timesReal(h));
	}

	/**
	 * Performs mean curvature flow on the input mesh with timestep h.
	 * @method module:Projects.MeanCurvatureFlow#integrate
	 * @param {number} h The timestep.
	 */
	integrate(h) {
		let vertices = this.geometry.mesh.vertices;
		let x = DenseMatrix.zeros(this.geometry.mesh.vertices.length, 1);
		let y = DenseMatrix.zeros(this.geometry.mesh.vertices.length, 1);
		let z = DenseMatrix.zeros(this.geometry.mesh.vertices.length, 1);
		for (let i in this.geometry.positions) {
			x.set(this.geometry.positions[i].x, this.vertexIndex[i], 0);
			y.set(this.geometry.positions[i].y, this.vertexIndex[i], 0);
			z.set(this.geometry.positions[i].z, this.vertexIndex[i], 0);
		}
		
		let M = this.geometry.massMatrix(this.vertexIndex);
		let A = this.buildFlowOperator(M, h);
		let llt = A.chol();
		x = llt.solvePositiveDefinite(M.timesDense(x));
		y = llt.solvePositiveDefinite(M.timesDense(y));
		z = llt.solvePositiveDefinite(M.timesDense(z));

		for (let i in this.geometry.positions) {
			this.geometry.positions[i].x = x.get(this.vertexIndex[i], 0);
			this.geometry.positions[i].y = y.get(this.vertexIndex[i], 0);
			this.geometry.positions[i].z = z.get(this.vertexIndex[i], 0);
		}

		// center mesh positions around origin
		normalize(this.geometry.positions, vertices, false);
	}
}
