"use strict";

/**
 * This class contains methods to build common {@link https://cs.cmu.edu/~kmcrane/Projects/DDG/paper.pdf discrete exterior calculus} operators.
 * @memberof module:Core
 */
class DEC {
	/**
	 * Builds a sparse diagonal matrix encoding the Hodge operator on 0-forms.
	 * By convention, the area of a vertex is 1.
	 * @static
	 * @param {module:Core.Geometry} geometry The geometry of a mesh.
	 * @param {Object} vertexIndex A dictionary mapping each vertex of a mesh to a unique index.
	 * @returns {module:LinearAlgebra.SparseMatrix}
	 */
	static buildHodgeStar0Form(geometry, vertexIndex) {
		let vertices = geometry.mesh.vertices;
		let T = new Triplet(vertices.length, vertices.length);
		for (let i of Object.keys(vertexIndex)) {
			T.addEntry(geometry.barycentricDualArea(vertices[i]),
				vertexIndex[i], vertexIndex[i]);
		}
		return SparseMatrix.fromTriplet(T);
	}

	/**
	 * Builds a sparse diagonal matrix encoding the Hodge operator on 1-forms.
	 * @static
	 * @param {module:Core.Geometry} geometry The geometry of a mesh.
	 * @param {Object} edgeIndex A dictionary mapping each edge of a mesh to a unique index.
	 * @returns {module:LinearAlgebra.SparseMatrix}
	 */
	static buildHodgeStar1Form(geometry, edgeIndex) {
		let edges = geometry.mesh.edges;
		let T = new Triplet(edges.length, edges.length);
		for (let i of Object.keys(edgeIndex)) {
			T.addEntry((geometry.cotan(edges[i].halfedge) +
				geometry.cotan(edges[i].halfedge.twin)) / 2,
				edgeIndex[i], edgeIndex[i]);
		}
		return SparseMatrix.fromTriplet(T);
	}

	/**
	 * Builds a sparse diagonal matrix encoding the Hodge operator on 2-forms.
	 * By convention, the area of a vertex is 1.
	 * @static
	 * @param {module:Core.Geometry} geometry The geometry of a mesh.
	 * @param {Object} faceIndex A dictionary mapping each face of a mesh to a unique index.
	 * @returns {module:LinearAlgebra.SparseMatrix}
	 */
	static buildHodgeStar2Form(geometry, faceIndex) {
		let faces = geometry.mesh.faces;
		let T = new Triplet(faces.length, faces.length);
		for (let i of Object.keys(faceIndex)) {
			T.addEntry(1 / geometry.area(faces[i]), faceIndex[i], faceIndex[i]);
		}
		return SparseMatrix.fromTriplet(T);
	}

	/**
	 * Builds a sparse matrix encoding the exterior derivative on 0-forms.
	 * @static
	 * @param {module:Core.Geometry} geometry The geometry of a mesh.
	 * @param {Object} edgeIndex A dictionary mapping each edge of a mesh to a unique index.
	 * @param {Object} vertexIndex A dictionary mapping each vertex of a mesh to a unique index.
	 * @returns {module:LinearAlgebra.SparseMatrix}
	 */
	static buildExteriorDerivative0Form(geometry, edgeIndex, vertexIndex) {
		let edges = geometry.mesh.edges;
		let vertices = geometry.mesh.vertices;
		let T = new Triplet(edges.length, vertices.length);
		for (let i of Object.keys(edgeIndex)) {
			T.addEntry(-1, edgeIndex[i], edges[i].halfedge.vertex.index);
			T.addEntry(1, edgeIndex[i], edges[i].halfedge.twin.vertex.index);
		}
		return SparseMatrix.fromTriplet(T);
	}

	/**
	 * Builds a sparse matrix encoding the exterior derivative on 1-forms.
	 * @static
	 * @param {module:Core.Geometry} geometry The geometry of a mesh.
	 * @param {Object} faceIndex A dictionary mapping each face of a mesh to a unique index.
	 * @param {Object} edgeIndex A dictionary mapping each edge of a mesh to a unique index.
	 * @returns {module:LinearAlgebra.SparseMatrix}
	 */
	static buildExteriorDerivative1Form(geometry, faceIndex, edgeIndex) {
		let faces = geometry.mesh.faces;
		let edges = geometry.mesh.edges;
		let T = new Triplet(faces.length, edges.length);
		for (let i of Object.keys(faceIndex)) {
			let halfedge = faces[i].halfedge;
			for (let j = 0; j < 3; j++) {
				T.addEntry(Math.pow(-1, halfedge.vertex != halfedge.edge.halfedge.vertex),
					faceIndex[i], halfedge.edge.index);
				halfedge = halfedge.next;
			}
		}
		return SparseMatrix.fromTriplet(T);
	}
}
