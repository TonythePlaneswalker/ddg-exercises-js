"use strict";

class SpectralConformalParameterization {
	/**
	 * This class implements the {@link https://cs.cmu.edu/~kmcrane/Projects/DDG/paper.pdf spectral conformal parameterization} algorithm to flatten
	 * surface meshes with boundaries conformally.
	 * @constructor module:Projects.SpectralConformalParameterization
	 * @param {module:Core.Geometry} geometry The input geometry of the mesh this class acts on.
	 * @property {module:Core.Geometry} geometry The input geometry of the mesh this class acts on.
	 * @property {Object} vertexIndex A dictionary mapping each vertex of the input mesh to a unique index.
	 */
	constructor(geometry) {
		this.geometry = geometry;
		this.vertexIndex = indexElements(geometry.mesh.vertices);
	}

	/**
	 * Builds the complex conformal energy matrix EC = ED - A.
	 * @private
	 * @method module:Projects.SpectralConformalParameterization#buildConformalEnergy
	 * @returns {module:LinearAlgebra.ComplexSparseMatrix}
	 */
	buildConformalEnergy() {
		let ED = this.geometry.complexLaplaceMatrix(this.vertexIndex);
		ED.scaleBy(new Complex(0.5, 0));

		let v = this.geometry.mesh.vertices.length;
		let T = new ComplexTriplet(v, v);
		for (let f of this.geometry.mesh.boundaries) {
			let h = f.halfedge;
			do {
				console.log(h.vertex.index, h.twin.vertex.index, h.onBoundary);
				T.addEntry(new Complex(0, 0.25), h.vertex.index, h.twin.vertex.index);
				T.addEntry(new Complex(0, -0.25), h.twin.vertex.index, h.vertex.index);
				h = h.next;
			} while (h != f.halfedge);
		}
		let A = ComplexSparseMatrix.fromTriplet(T);

		return ED.minus(A);
	}

	/**
	 * Flattens the input surface mesh with 1 or more boundaries conformally.
	 * @method module:Projects.SpectralConformalParameterization#flatten
	 * @returns {Object} A dictionary mapping each vertex to a vector of planar coordinates.
	 */
	flatten() {
		let vertices = this.geometry.mesh.vertices;
		let flattening = {};
		let A = this.buildConformalEnergy();
		let x = Solvers.solveInversePowerMethod(A);
		for (let v of vertices) {
			flattening[v] = new Vector(x.get(v.index, 0).re, x.get(v.index, 0).im, 0);
		}

		// normalize flattening
		normalize(flattening, vertices);

		return flattening;
	}
}
